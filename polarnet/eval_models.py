from typing import Tuple, Dict, List

import os
import numpy as np
import itertools
from tqdm import tqdm
import copy
from pathlib import Path
import jsonlines
import json
import tap
import open3d as o3d
from filelock import FileLock
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.multiprocessing as mp

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


from polarnet.utils.misc import set_random_seed
from polarnet.config.default import get_config
from polarnet.core.actioner import BaseActioner
from polarnet.core.environments import RLBenchEnv
from polarnet.models.pcd_unet import PointCloudUNet

from polarnet.config.constants import get_workspace
from polarnet.utils.coord_transforms import discrete_euler_to_quaternion


class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'  # cpu, cuda

    eval_train_split: bool = False
    microstep_data_dir: Path = None
    microstep_outname: str = 'microsteps'

    seed: int = 100  # seed for RLBench
    num_demos: int = 500

    headless: bool = False
    max_tries: int = 10
    save_image: bool = False

    checkpoint: str = None # dir evaluate all, path evaluate one
    num_workers: int = 1
    start_ckpt_step: int = 0

    taskvars: str = None

    save_obs_outs_dir: str = None
    record_video: bool = False
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    taskvar_episodes_file: str = None

    ignore_existed_taskvars: bool = False

    real_robot: bool = False

    cam_rand_factor: float = 0.0
    image_size: List[int] = [128, 128]
    instr_embed_file: str = None

class Actioner(BaseActioner):
    def __init__(self, args) -> None:
        self.args = args
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        self.WORKSPACE = get_workspace(real_robot=args.real_robot)

        self.device = torch.device(args.device)

        config = get_config(args.exp_config, args.remained_args)
        self.config = config
        self.config.defrost()

        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint

        self.gripper_channel = self.config.MODEL.gripper_channel = \
            self.config.MODEL.get('gripper_channel', False)
        self.model = PointCloudUNet(**config.MODEL)
        
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.config.freeze()

        self.use_instr_embed = config.MODEL.use_instr_embed
        if type(config.DATASET.taskvars) is str:
            config.DATASET.taskvars = config.DATASET.taskvars.split(',')
        self.taskvars = config.DATASET.taskvars

        if self.use_instr_embed != 'none':
            assert config.DATASET.instr_embed_file is not None
            instr_embed_file = config.DATASET.instr_embed_file if args.instr_embed_file is None else args.instr_embed_file
            self.lmdb_instr_env = lmdb.open(
                instr_embed_file, readonly=True, lock=False)
            self.lmdb_instr_txn = self.lmdb_instr_env.begin()
            self.memory = {'instr_embeds': {}}
        else:
            self.lmdb_instr_env = None

        self.tasks_use_table_surface = set(json.load(open('assets/tasks_use_table_surface.json')))

    def __exit__(self):
        self.lmdb_instr_env.close()

    def get_taskvar_instr_embeds(self, taskvar, step_id):
        instr_embeds = None
        if taskvar in self.memory['instr_embeds']:
            instr_embeds = self.memory['instr_embeds'][taskvar]

        if instr_embeds is None or step_id == 0:
            instr_embeds = self.lmdb_instr_txn.get(taskvar.encode('ascii'))
            instr_embeds = msgpack.unpackb(instr_embeds)
            instr_embeds = [torch.from_numpy(x).float() for x in instr_embeds]
            instr_idx = np.random.randint(len(instr_embeds))

            instr_embeds = instr_embeds[instr_idx]

            if self.use_instr_embed == 'avg':
                instr_embeds = torch.mean(instr_embeds, 0, keepdim=True)
            elif self.use_instr_embed == 'last':
                instr_embeds = instr_embeds[-1:]
            self.memory['instr_embeds'][taskvar] = instr_embeds
        return instr_embeds  # (num_ttokens, dim)
    
    def process_point_clouds(self, rgb, pc, gripper_pose=None, task=None, add_pcd_noises=False, pcd_noises_std=0.01):
        pc = pc.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)

        X_BBOX = self.WORKSPACE['X_BBOX']
        Y_BBOX = self.WORKSPACE['Y_BBOX']
        Z_BBOX = self.WORKSPACE['Z_BBOX']
        TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']

        if self.config.DATASET.pc_space in ['workspace', 'workspace_on_table']:
            masks = (pc[:, 0] > X_BBOX[0]) & (pc[:, 0] < X_BBOX[1]) & \
                    (pc[:, 1] > Y_BBOX[0]) & (pc[:, 1] < Y_BBOX[1]) & \
                    (pc[:, 2] > Z_BBOX[0]) & (pc[:, 2] < Z_BBOX[1])
            if self.config.DATASET.pc_space == 'workspace_on_table' and \
                task not in self.tasks_use_table_surface:
                masks = masks & (pc[:, 2] > TABLE_HEIGHT)
            if np.sum(masks) == 0: # deal with exception
                masks[:] = True
            rgb = rgb[masks]
            pc = pc[masks]

        # pcd center and radius
        if self.config.DATASET.pc_center == 'point':
            pc_center = np.mean(pc, 0)
        elif self.config.DATASET.pc_center == 'gripper':
            pc_center = gripper_pose[:3]
        
        if self.config.DATASET.pc_radius_norm:
            pc_radius = np.max(np.sqrt(np.sum((pc - pc_center)**2, 1)), keepdims=True)
        else:
            pc_radius = np.ones((1, ), dtype=np.float32)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb  / 255.)

        if self.config.DATASET.voxel_size is not None and self.config.DATASET.voxel_size > 0:
            downpcd, _, idxs = pcd.voxel_down_sample_and_trace(
                self.config.DATASET.voxel_size, np.min(pc, 0), np.max(pc, 0)
            )
        else:
            downpcd = pcd

        if getattr(self.config.DATASET, 'remove_pcd_outliers', False):    # TODO: adhoc
           downpcd, outlier_masks = downpcd.remove_radius_outlier(nb_points=32, radius=0.05)

        new_rgb = np.asarray(downpcd.colors) * 2 - 1 # (-1, 1)
        new_pos = np.asarray(downpcd.points)

        if add_pcd_noises:
            new_pos = new_pos + np.random.randn(*new_pos.shape) * pcd_noises_std

        new_ft = (new_pos - pc_center) / pc_radius
        if self.config.DATASET.use_color:
            new_ft = np.concatenate([new_ft, new_rgb], axis=-1)

        if self.config.DATASET.use_normal:
            downpcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.02 if self.config.DATASET.voxel_size is None else self.config.DATASET.voxel_size*2, 
                    max_nn=30
                )
            )
            new_ft = np.concatenate(
                [new_ft, np.asarray(downpcd.normals)], axis=-1
            )
        if self.config.DATASET.use_height:
            heights = np.asarray(downpcd.points)[:, -1]
            heights = heights - TABLE_HEIGHT
            new_ft = np.concatenate(
                [new_ft, heights[:, None]], axis=-1
            )

        ridx = np.random.permutation(len(new_rgb))[:self.config.DATASET.npoints]
        if len(ridx) < self.config.DATASET.npoints:
            ridx = np.concatenate(
                [ridx] * np.ceil(self.config.DATASET.npoints/len(ridx)).astype(np.int32), axis=0
            )[:self.config.DATASET.npoints]
        new_pos = new_pos[ridx]
        new_ft = new_ft[ridx].transpose(1, 0)

        return new_pos, new_ft, pc_center, pc_radius

    def preprocess_obs(self, taskvar_id, step_id, obs, add_pcd_noises=False, pcd_noises_std=0.01):
        taskvar = self.taskvars[taskvar_id]
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        pcd = np.stack(obs['pc'], 0)  # (N, H, W, C)
        
        camera_ids = self.config.DATASET.get('camera_ids', None)
        if camera_ids is not None:
            rgb = rgb[camera_ids]
            pcd = pcd[camera_ids]

        if 'PointCloud' in self.config.MODEL.model_class:
            pos, ft, pc_center, pc_radius = self.process_point_clouds(
                rgb, pcd, gripper_pose=obs['gripper'], task=taskvar.split('+')[0],
                add_pcd_noises=add_pcd_noises, pcd_noises_std=pcd_noises_std
            )
            batch = {
                'poses': torch.from_numpy(pos).float().unsqueeze(0),
                'fts': torch.from_numpy(ft).float().unsqueeze(0),
                'pc_centers': torch.from_numpy(pc_center).float().unsqueeze(0),
                'pc_radii': torch.from_numpy(pc_radius).float().unsqueeze(0),
            }

        else:
            rgb = torch.from_numpy(rgb).float().permute(0, 3, 1, 2)
            # normalise to [-1, 1]
            rgb = 2 * (rgb / 255.0 - 0.5)

            pcd = torch.from_numpy(pcd).float().permute(0, 3, 1, 2)
            if self.gripper_channel:
                gripper_imgs = torch.from_numpy(
                    obs["gripper_imgs"]).float()  # (N, 1, H, W)
                rgb = torch.cat([rgb, gripper_imgs], dim=1)
            
            batch = {
                'rgbs': rgb.unsqueeze(0),
                'pcds': pcd.unsqueeze(0),
            }

        batch['step_ids'] = torch.LongTensor([step_id])
        batch['taskvar_ids'] = torch.LongTensor([taskvar_id])
        if self.config.MODEL.use_prev_action:
            batch['prev_actions'] = torch.from_numpy(obs['gripper']).float().unsqueeze(0)
        
        if self.use_instr_embed != 'none':
            batch['instr_embeds'] = self.get_taskvar_instr_embeds(
                taskvar, step_id).unsqueeze(0)
            batch['txt_masks'] = torch.ones(
                1, batch['instr_embeds'].size(1)).long()

        return batch

    def predict(self, taskvar_id, step_id, obs_state_dict, episode_id=None, add_pcd_noises=False, pcd_noises_std=0.01):
        batch = self.preprocess_obs(
            taskvar_id, step_id, obs_state_dict,
            add_pcd_noises=add_pcd_noises, pcd_noises_std=pcd_noises_std
        )
        with torch.no_grad():
            action = self.model(batch)[0]

        action[-1] = torch.sigmoid(action[-1])

        action = action.data.cpu().numpy()

        if self.config.MODEL.get('use_discrete_rot', False):
            rot_logits = action[3:-1].reshape(3, -1)
            rot_quat = discrete_euler_to_quaternion(
                np.argmax(rot_logits, axis=1), self.config.MODEL.rot_resolution
            )
            action = np.concatenate(
                [action[:3], rot_quat, action[-1:]], axis=0
            )

        out = {
            'action': action
        }

        if self.args.save_obs_outs_dir is not None:
            np.save(
                os.path.join(self.args.save_obs_outs_dir, f'{taskvar_id}-{episode_id}-{step_id}.npy'),
                {
                    'batch': batch,
                    'obs': obs_state_dict,
                    'action': action
                }
            )

        return out


def write_to_file(filepath, data):
    lock = FileLock(filepath+'.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)

def evaluate_keysteps(args_tuple):
    args, checkpoint, taskvar = args_tuple

    args = copy.deepcopy(args)
    if checkpoint is not None:
        args.checkpoint = checkpoint

    set_random_seed(args.seed)

    actioner = Actioner(args)

    config = actioner.config
    config.defrost()
    actioner.taskvars = config.DATASET.taskvars = [taskvar]
    config.freeze()

    if args.eval_train_split:
        microstep_data_dir = Path(
            config.DATASET.data_dir.replace('keysteps', 'microsteps'))
        pred_dir = os.path.join(config.output_dir, 'preds', 'train')
    else:
        if args.microstep_data_dir is not None:
            microstep_data_dir = args.microstep_data_dir
            pred_dir = os.path.join(config.output_dir, 'preds', args.microstep_outname)
        else:
            microstep_data_dir = ''
            pred_dir = os.path.join(config.output_dir, 'preds', f'seed{args.seed}')
    if args.cam_rand_factor > 0:
        pred_dir = '%s-cam_rand_factor%.1f' % (pred_dir, args.cam_rand_factor)
    os.makedirs(pred_dir, exist_ok=True)

    env = RLBenchEnv(
        data_path=microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=config.DATASET.cameras,
        headless=args.headless,
        gripper_pose=config.MODEL.gripper_channel,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )

    outfile = os.path.join(pred_dir, 'results.jsonl')

    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], '%s+%d'%(item['task'], item['variation'])))

    # BUG: in headless mode, run multiple taskvars can have problems
    for taskvar_id, taskvar in enumerate(actioner.taskvars):
        if (not args.ignore_existed_taskvars) and ((args.checkpoint, taskvar) in existed_data):
            continue

        task_str, variation = taskvar.split('+')
        variation = int(variation)

        if args.eval_train_split or args.microstep_data_dir is not None:
            episodes_dir = microstep_data_dir / task_str / \
                f"variation{variation}" / "episodes"
            if not os.path.exists(str(episodes_dir)):
                print('taskvar', taskvar, 'not exists')
                continue
            demo_keys, demos = [], []
            for ep in tqdm(episodes_dir.glob('episode*')):
                if args.microstep_data_dir is not None:
                    episode_id = len(demo_keys)
                else:
                    episode_id = int(ep.stem[7:])
                demo = env.get_demo(task_str, variation, episode_id)
                demo_keys.append(f'episode{episode_id}')
                demos.append(demo)
                # if len(demos) > 1:
                #     break
            num_demos = len(demos)
        else:
            demo_keys = None
            demos = None
            if args.taskvar_episodes_file is None:
                num_demos = args.num_demos
            else:
                taskvar_episodes = json.load(open(args.taskvar_episodes_file, 'r'))
                num_demos = taskvar_episodes[taskvar]

        success_rate = env.evaluate(
            taskvar_id,
            task_str,
            actioner=actioner,
            max_episodes=config.MODEL.max_steps,
            variation=variation,
            num_demos=num_demos,
            demos=demos,
            demo_keys=demo_keys,
            log_dir=Path(pred_dir),
            max_tries=args.max_tries,
            save_image=args.save_image,
            record_video=args.record_video,
            include_robot_cameras=(not args.not_include_robot_cameras),
            video_rotate_cam=args.video_rotate_cam,
            video_resolution=args.video_resolution,
        )

        print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
        write_to_file(
            outfile,
            {
                'checkpoint': config.checkpoint,
                'task': task_str, 'variation': variation,
                'num_demos': num_demos, 'sr': success_rate
            }
        )


def main(args):

    config = get_config(args.exp_config, args.remained_args)
    if args.taskvars is not None:
        taskvars = args.taskvars.split(',')
    else:
        if len(config.DATASET.taskvars) == 1 and os.path.exists(config.DATASET.taskvars[0]):
            taskvars = list(json.load(open(config.DATASET.taskvars[0], 'r')).keys())
            taskvars.sort()
        else:
            taskvars = config.DATASET.taskvars

    if args.checkpoint is not None and os.path.isdir(args.checkpoint):
        checkpoints = os.listdir(args.checkpoint)
        checkpoints.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        checkpoints = [os.path.join(args.checkpoint, x) for x in checkpoints if int(x.split('.')[0].split('_')[-1]) >= args.start_ckpt_step]
    else:
        checkpoints = [args.checkpoint]

    num_jobs = len(checkpoints) * len(taskvars)
    num_workers = min(args.num_workers, num_jobs)
    
    if num_jobs == 1:
        evaluate_keysteps((args, checkpoints[0], taskvars[0]))

    else:
        job_args = []
        for ckpt in checkpoints:
            for taskvar in taskvars:
                job_args.append((args, ckpt, taskvar))

        mp.set_start_method('spawn')

        pool = mp.Pool(num_workers)
        pool.map(evaluate_keysteps, job_args)
        pool.close()
        pool.join()


if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    main(args)

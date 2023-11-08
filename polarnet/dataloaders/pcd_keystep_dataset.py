from typing import List, Dict, Optional

import os
import numpy as np
import copy
import json
import open3d as o3d
from PIL import Image
import einops
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from polarnet.utils.utils import get_assets_dir
from polarnet.utils.ops import pad_tensors, gen_seq_masks
from polarnet.dataloaders.keystep_dataset import KeystepDataset
from polarnet.config.constants import get_workspace

from polarnet.utils.coord_transforms import quaternion_to_discrete_euler


def action_rot_quat_to_euler(action, resolution):
    pos = action[:3]
    quat = action[3:7]
    open = action[7]
    rot_disc = quaternion_to_discrete_euler(quat, resolution)
    return np.concatenate([pos, rot_disc, [open]]).astype(np.float32)

def random_shift_pcd_and_action(pcd, action, shift_range, shift=None):
    '''
    pcd: (npoints, 3) or (T, 3, npoints)
    action: (8) or (T, 8)
    shift_range: float
    '''
    if shift is None:
        shift = np.random.uniform(-shift_range, shift_range, size=(3, ))

    if len(pcd.shape) == 2:
        pcd = pcd + shift
        action[:3] += shift

    elif len(pcd.shape) == 3:
        pcd = pcd + shift[None, :, None]
        action[..., :3] += shift[None, :]

    return pcd, action

def random_rotate_pcd_and_action(pcd, action, rot_range, rot=None):
    '''
    pcd: (npoints, 3) or (T, 3, npoints)
    action: (8) or (T, 8)
    shift_range: float
    '''
    if rot is None:
        rot = np.random.uniform(-rot_range, rot_range)
    r = R.from_euler('z', rot, degrees=True)

    if len(pcd.shape) == 2:
        pcd = r.apply(pcd)
        action[:3] = r.apply(action[:3])
        
        a_ori = R.from_quat(action[3:7])
        a_new = r * a_ori
        action[3:7] = a_new.as_quat()

    elif len(pcd.shape) == 3:
        pos_ori = einops.rearrange(pcd, 't c n -> (t n) c')
        pos_new = r.apply(pos_ori)
        pcd = einops.rearrange(pos_new, '(t n) c -> t c n', t=pcd.shape[0], n=pcd.shape[2])
        action[..., :3] = r.apply(action[..., :3])
        
        a_ori = R.from_quat(action[..., 3:7])
        a_new = r * a_ori
        action[..., 3:7] = a_new.as_quat()

    return pcd, action

class PCDKeystepDataset(KeystepDataset):
    def __init__(
            self, data_dir, taskvars, instr_embed_file=None, 
            gripper_channel=False, camera_ids=None, cameras=..., use_instr_embed='none', 
            is_training=False, in_memory=False, 
            voxel_size=0.01, npoints=2048, use_color=True,
            use_normal=True, use_height=True, pc_space='none',
            color_drop=0, pc_center='point', pc_radius_norm=True, **kwargs
        ):
        '''
        - pc_space:
            - none: no filter points
            - workspace: filter points inside x_bbox, y_bbox, and z_bbox
            - workspace_on_table: filter points inside 3 bboxes and above the table height
        '''
        super().__init__(
            data_dir, taskvars, instr_embed_file, gripper_channel, camera_ids, 
            cameras, use_instr_embed, is_training, in_memory, **kwargs
        )
        self.voxel_size = voxel_size
        self.npoints = npoints
        self.use_normal = use_normal
        self.use_height = use_height
        self.use_color = use_color
        self.color_drop = color_drop
        self.pc_space = pc_space
        self.pc_center = pc_center
        self.pc_radius_norm = pc_radius_norm
        self.rgb_augment = kwargs.get('rgb_augment', False)
        self.max_steps_per_episode = kwargs.get('max_steps_per_episode', None)
        self.add_pcd_noises = kwargs.get('add_pcd_noises', False)
        self.pcd_noises_std = kwargs.get('pcd_noises_std', 0.01)
        self.remove_pcd_outliers = kwargs.get('remove_pcd_outliers', False)
        self.WORKSPACE = get_workspace(real_robot=kwargs.get('real_robot', False))
        self.use_discrete_rot = kwargs.get('use_discrete_rot', False)
        self.rot_resolution = kwargs.get('rot_resolution', 5)
        self.aug_shift_pcd = kwargs.get('aug_shift_pcd', 0.0)       # shift pcd by x meters
        self.aug_rotate_pcd = kwargs.get('aug_rotate_pcd', 0.0)     # rotate pcd by x degrees

        assert self.pc_space in ['none', 'workspace', 'workspace_on_table']
        assert self.pc_center in ['gripper', 'point']

        self.tasks_with_color = set(json.load(open(f'{get_assets_dir()}/tasks_with_color.json')))
        self.tasks_use_table_surface = set(json.load(open(f'{get_assets_dir()}/tasks_use_table_surface.json')))


    def get_taskvar_episode(self, taskvar_idx, episode_key):
        if self.in_memory:
            mem_key = f'taskvar{taskvar_idx}'
            if episode_key in self.memory[mem_key]:
                return self.memory[mem_key][episode_key]
        
        task = self.taskvars[taskvar_idx].split('+')[0]

        value = self.lmdb_txns[taskvar_idx].get(episode_key)
        value = msgpack.unpackb(value)

        # The last one is the stop observation: (T, N, H, W, 3)
        num_steps, num_cameras, im_height, im_width, _  = value['rgb'].shape

        rgbs = value['rgb'].reshape(num_steps, -1, 3) / 255. # (T, N*H*W, C), [0, 1]
        pcs = np.array(value['pc'].reshape(num_steps, -1, 3))

        poses, fts, pc_centers, pc_radii, actions = [], [], [], [], []
        t = 0
        random_shift, random_rot = None, None
        if self.aug_rotate_pcd > 0:
            random_rot = np.random.uniform(-self.aug_rotate_pcd, self.aug_rotate_pcd)
        if self.aug_shift_pcd > 0:
            random_shift = np.random.uniform(-self.aug_shift_pcd, self.aug_shift_pcd, size=(3, ))
            
        for rgb, pc, gripper_pose in zip(rgbs, pcs, value['action']):
            new_pos, new_ft, pc_center, pc_radius, action = self.process_point_clouds(
                rgb, pc, gripper_pose=gripper_pose, task=task, 
                random_shift=random_shift, random_rot=random_rot
            )
            poses.append(new_pos)
            fts.append(new_ft)
            pc_centers.append(pc_center)
            pc_radii.append(pc_radius)
            actions.append(action)
            t += 1

        value['fts'] = fts[:-1]
        value['poses'] = poses[:-1]
        value['pc_centers'] = np.stack(pc_centers[:-1], 0)
        value['pc_radii'] = np.stack(pc_radii[:-1], 0)
        value['action'] = np.stack(actions, 0)
        del value['pc']

        if self.in_memory:
            self.memory[mem_key][episode_key] = value
        return value
    
    def process_point_clouds(
            self, rgb, pc, gripper_pose=None, task=None, 
            random_shift=None, random_rot=None
        ):
        gripper_pose = copy.deepcopy(gripper_pose)

        X_BBOX = self.WORKSPACE['X_BBOX']
        Y_BBOX = self.WORKSPACE['Y_BBOX']
        Z_BBOX = self.WORKSPACE['Z_BBOX']
        TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']

        if self.pc_space in ['workspace', 'workspace_on_table']:
            masks = (pc[:, 0] > X_BBOX[0]) & (pc[:, 0] < X_BBOX[1]) & \
                    (pc[:, 1] > Y_BBOX[0]) & (pc[:, 1] < Y_BBOX[1]) & \
                    (pc[:, 2] > Z_BBOX[0]) & (pc[:, 2] < Z_BBOX[1])
            if self.pc_space == 'workspace_on_table' and task not in self.tasks_use_table_surface:
                masks = masks & (pc[:, 2] > TABLE_HEIGHT)
            rgb = rgb[masks]
            pc = pc[masks]

        if self.aug_shift_pcd > 0:
            pc, gripper_pose = random_shift_pcd_and_action(
                pc, gripper_pose, self.aug_shift_pcd, shift=random_shift
            )

        if self.aug_rotate_pcd > 0:
            pc, gripper_pose = random_rotate_pcd_and_action(
                pc, gripper_pose, self.aug_rotate_pcd, rot=random_rot
            )

        # pcd center and radius
        if self.pc_center == 'point':
            pc_center = np.mean(pc, 0)
        elif self.pc_center == 'gripper':
            pc_center = gripper_pose[:3]
            
        if self.pc_radius_norm:
            pc_radius = np.max(np.sqrt(np.sum((pc - pc_center)**2, 1)), keepdims=True)
        else:
            pc_radius = np.ones((1, ), dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        if self.voxel_size is not None and self.voxel_size > 0:
            downpcd, _, idxs = pcd.voxel_down_sample_and_trace(self.voxel_size, np.min(pc, 0), np.max(pc, 0))
        else:
            downpcd = pcd

        if self.remove_pcd_outliers:    # TODO: adhoc
           downpcd, outlier_masks = downpcd.remove_radius_outlier(nb_points=32, radius=0.05)

        new_rgb = np.asarray(downpcd.colors) * 2 - 1 # (-1, 1)
        new_pos = np.asarray(downpcd.points)

        if self.add_pcd_noises:
            new_pos = new_pos + np.random.randn(*new_pos.shape) * self.pcd_noises_std

        # normalized point clouds
        new_ft = (new_pos - pc_center) / pc_radius
        if self.use_color:
            new_ft = np.concatenate([new_ft, new_rgb], axis=-1)
        if self.use_normal:
            downpcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=getattr(self, 'voxel_size', 0.01)*2, max_nn=30
            ))
            new_ft = np.concatenate(
                [new_ft, np.asarray(downpcd.normals)], axis=-1
            )
        if self.use_height:
            heights = np.asarray(downpcd.points)[:, -1]
            heights = heights - TABLE_HEIGHT
            if random_shift is not None:
                heights = heights - random_shift[-1]
            new_ft = np.concatenate(
                [new_ft, heights[:, None]], axis=-1
            )

        return new_pos, new_ft, pc_center, pc_radius, gripper_pose

        
    def __getitem__(self, idx):
        taskvar_idx, episode_key = self.episode_ids[idx]

        value = self.get_taskvar_episode(taskvar_idx, episode_key)

        poses, fts = [], []
        for pos, ft in zip(value['poses'], value['fts']):
            ridx = np.random.permutation(len(pos))[:self.npoints]
            if len(ridx) < self.npoints:
                ridx = np.concatenate(
                    [ridx] * np.ceil(self.npoints/len(ridx)).astype(np.int32), axis=0
                )[:self.npoints]
            poses.append(pos[ridx])
            fts.append(ft[ridx])
        poses = np.stack(poses, 0)  # (T, npoints, 3)
        fts = np.stack(fts, 0).transpose(0, 2, 1)      # (T, dim_ft, npoints)

        if self.use_color and self.taskvars[taskvar_idx].split('+')[0] not in self.tasks_with_color \
            and self.color_drop > 0 and np.random.rand() < self.color_drop:
            fts[:, 3:6] = 0

        outs = {
            'fts': torch.from_numpy(fts).float(),
            'pc_centers': torch.from_numpy(value['pc_centers']).float(),
            'pc_radii': torch.from_numpy(value['pc_radii']).float()
        }
        
        num_steps = len(outs['fts'])

        actions = value['action']
        if self.use_discrete_rot:
            actions = np.stack([action_rot_quat_to_euler(a, self.rot_resolution) for a in actions], 0)

        outs['step_ids'] = torch.arange(0, num_steps).long()
        outs['prev_actions'] = torch.from_numpy(actions[:-1])
        outs['actions'] = torch.from_numpy(actions[1:])
        outs['episode_ids'] = episode_key.decode('ascii')
        outs['taskvars'] = self.taskvars[taskvar_idx]
        outs['taskvar_ids'] = taskvar_idx

        if self.exclude_overlength_episodes is not None:
            for key in ['fts', 'pc_centers', 'pc_radii', 'rgbs', 'step_ids', 'prev_actions', 'actions']:
                if key in outs:
                    outs[key] = outs[key][:self.exclude_overlength_episodes]
            num_steps = len(outs['fts'])

        if (self.max_steps_per_episode is not None) and num_steps > self.max_steps_per_episode :
            sidx = np.random.randint(
                0, num_steps - self.max_steps_per_episode + 1
            )
            for key in ['fts', 'pc_centers', 'pc_radii', 'rgbs', 'step_ids', 'prev_actions', 'actions']:
                if key in outs:
                    outs[key] = outs[key][sidx:sidx+self.max_steps_per_episode]

        if self.use_instr_embed != 'none':
            outs['instr_embeds'] = self.get_taskvar_instr_embeds(outs['taskvars'])

        return outs


class ProcessedPCDKeystepDataset(PCDKeystepDataset):

    def get_taskvar_episode(self, taskvar_idx, episode_key):
        if self.in_memory:
            mem_key = f'taskvar{taskvar_idx}'
            if episode_key in self.memory[mem_key]:
                return self.memory[mem_key][episode_key]
        
        value = self.lmdb_txns[taskvar_idx].get(episode_key)
        value = msgpack.unpackb(value)
        
        if self.aug_shift_pcd > 0 or self.aug_rotate_pcd > 0:

            random_shift = np.random.uniform(-self.aug_shift_pcd, self.aug_shift_pcd, size=(3, ))
            random_rot = np.random.uniform(-self.aug_rotate_pcd, self.aug_rotate_pcd)

            pc_fts, actions, pc_centers, pc_radii = [], [], [], []
            for k in range(len(value['pc_fts'])):
                pcd = value['pc_fts'][k][:, :3]    # (npoints, 3)
                pcd = pcd * value['pc_radii'][k] + value['pc_centers'][k]
                action = copy.deepcopy(value['actions'][k])      # (8)

                if self.aug_shift_pcd > 0:
                    pcd, action = random_shift_pcd_and_action(
                        pcd, action, self.aug_shift_pcd, shift=random_shift
                    )
                if self.aug_rotate_pcd > 0:
                    pcd, action = random_rotate_pcd_and_action(
                        pcd, action, self.aug_rotate_pcd, rot=random_rot
                    )

                if self.pc_center == 'point':
                    pc_center = np.mean(pcd, -1)
                elif self.pc_center == 'gripper':
                    pc_center = action[:3]

                if self.pc_radius_norm:
                    pc_radius = np.max(np.sqrt(np.sum((pcd - pc_center)**2, 1)), keepdims=True)
                else:
                    pc_radius = np.ones((1, ), dtype=np.float32)

                pcd = (pcd - pc_center) / pc_radius

                pc_fts.append(
                    np.concatenate([pcd, value['pc_fts'][k][:, 3:]], -1)
                )
                pc_centers.append(pc_center)
                pc_radii.append(pc_radius)
                actions.append(action)

            value['pc_fts'] = pc_fts
            value['pc_centers'] = np.stack(pc_centers, 0)
            value['pc_radii'] = np.stack(pc_radii, 0)
            value['actions'] = np.stack(actions, 0)

        outs = {
            'fts': value['pc_fts'][:-1],
            'pc_centers': np.stack(value['pc_centers'][:-1], 0),
            'pc_radii': np.stack(value['pc_radii'][:-1], 0),
        }
        outs['action'] = value['actions']
        
        outs['poses'] = []
        for t in range(len(outs['fts'])):
            # (T, N, 3), (T, 3), (T, 3)
            outs['poses'].append(
                outs['fts'][t][:, :3] * outs['pc_radii'][t][None, :] + outs['pc_centers'][t][None, :]
            )

        if self.in_memory:
            self.memory[mem_key][episode_key] = outs
        return outs
    
    
def pcd_stepwise_collate_fn(data: List[Dict]):
    batch = {}
    
    for key in data[0].keys():
        if key == 'taskvar_ids':
            batch[key] = [
                torch.LongTensor([v['taskvar_ids']] * len(v['step_ids'])) for v in data
            ]
        elif key in ['taskvars', 'episode_ids']:
            batch[key] = sum([
                [v[key]] * len(v['step_ids']) for v in data
            ], [])
        elif key == 'instr_embeds':
            batch[key] = sum([
                [v['instr_embeds']] * len(v['step_ids']) for v in data
            ], [])
        else:
            batch[key] = [v[key] for v in data]

    for key in ['fts', 'pc_centers', 'pc_radii', 'rgbs',
                'taskvar_ids', 'step_ids', 'prev_actions', 'actions']:
        # e.g. fts: (B*T, C, npoints)
        if key in batch:
            batch[key] = torch.cat(batch[key], dim=0)

    if 'instr_embeds' in batch:
        num_ttokens = [len(x) for x in batch['instr_embeds']]
        batch['instr_embeds'] = pad_tensors(batch['instr_embeds'])
        batch['txt_masks'] = torch.from_numpy(gen_seq_masks(num_ttokens))

    return batch


def pcd_episode_collate_fn(data: List[Dict]):
    batch = {}

    for key in data[0].keys():
        batch[key] = [v[key] for v in data]

    batch['taskvar_ids'] = torch.LongTensor(batch['taskvar_ids'])
    if 'instr_embeds' in batch:
        num_ttokens = [len(x['instr_embeds']) for x in data]

    num_steps = [len(x['fts']) for x in data]
    max_steps = np.max(num_steps)

    # do not pad zeros (because we use batchnorm)
    pad_batch_fts = []
    for i, x in enumerate(batch['fts']):
        pad_batch_fts.append(
            torch.cat([x]*np.ceil(max_steps / num_steps[i]).astype(np.int32), 0)[:max_steps]
        )
    batch['fts'] = torch.stack(pad_batch_fts, 0)    # (B, max_steps, C, npoints)

    for key in ['pc_centers', 'pc_radii', 'rgbs',
                'step_ids', 'prev_actions', 'actions']:
        pad = 0
        if key == 'prev_actions':
            pad = 1
        # e.g. pc_centers: (B, T, 3)
        if key in batch:
            batch[key] = pad_tensors(batch[key], lens=num_steps, pad=pad)

    if 'instr_embeds' in batch:
        batch['instr_embeds'] = pad_tensors(batch['instr_embeds'], lens=num_ttokens)
        batch['txt_masks'] = torch.from_numpy(gen_seq_masks(num_ttokens))
    else:
        batch['txt_masks'] = torch.ones(len(num_steps), 1).bool()

    batch['step_masks'] = torch.from_numpy(gen_seq_masks(num_steps))
        
    return batch



if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    data_dir = 'data/train_dataset/keysteps/seed0'
    taskvars = ['pick_and_lift+0',
              'pick_up_cup+0',
              'put_knife_on_chopping_board+0',
              'put_money_in_safe+0',
              'push_button+0',
              'reach_target+0',
              'slide_block_to_target+0',
              'stack_wine+0',
              'take_money_out_safe+0',
              'take_umbrella_out_of_umbrella_stand+0']
    cameras = ['left_shoulder', 'right_shoulder', 'wrist']
    instr_embed_file = None
    instr_embed_file = 'data/train_dataset/taskvar_instrs/clip'

    dataset = PCDKeystepDataset(data_dir, taskvars, 
        instr_embed_file=instr_embed_file, 
        use_instr_embed='all', cameras=cameras, is_training=True,
        voxel_size=0.01, npoints=2048, use_color=True, color_drop=0.2,
        use_normal=True, use_height=True, pc_space='workspace_on_table',
        in_memory=True
    )

    data_loader = DataLoader(
        dataset, batch_size=16, 
        # collate_fn=pcd_stepwise_collate_fn,
        collate_fn=pcd_episode_collate_fn,
        shuffle=True,
    )

    print(len(dataset), len(data_loader))

    st = time.time()
    for epoch in range(1):
        for batch in data_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.size())

        et = time.time()
        print('epoch %d cost time: %.2fs' % (epoch, et - st))

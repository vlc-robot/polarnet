from typing import Tuple, Dict, List

import os
import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
import collections
import tap

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from polarnet.utils.keystep_detection import keypoint_discovery
from polarnet.utils.coord_transforms import convert_gripper_pose_world_to_image

from polarnet.core.environments import RLBenchEnv
from PIL import Image


class Arguments(tap.Tap):
    microstep_data_dir: Path = "data/train_dataset/microsteps/seed0"
    keystep_data_dir: Path = "data/train_dataset/keysteps/seed0"

    tasks: Tuple[str, ...] = ("pick_up_cup",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    
    max_variations: int = 1
    offset: int = 0


def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demo = env.get_demo(task_str, variation, episode)

    key_frames = keypoint_discovery(demo)
    key_frames.insert(0, 0)

    state_dict_ls = collections.defaultdict(list)
    for f in key_frames:
        state_dict = env.get_observation(demo._observations[f])
        for k, v in state_dict.items():
            if len(v) > 0:
                # rgb: (N: num_of_cameras, H, W, C); gripper: (7+1, )
                state_dict_ls[k].append(v) 

    for k, v in state_dict_ls.items():
        state_dict_ls[k] = np.stack(v, 0) # (T, N, H, W, C)

    action_ls = state_dict_ls['gripper'] # (T, 7+1)
    del state_dict_ls['gripper']

    return demo, key_frames, state_dict_ls, action_ls


def generate_keystep_dataset(args: Arguments):
    # load RLBench environment
    rlbench_env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=args.cameras,
    )

    tasks = args.tasks
    variations = range(args.offset, args.max_variations)

    for task_str, variation in itertools.product(tasks, variations):
        episodes_dir = args.microstep_data_dir / task_str / f"variation{variation}" / "episodes"

        output_dir = args.keystep_data_dir / f"{task_str}+{variation}"
        output_dir.mkdir(parents=True, exist_ok=True)

        lmdb_env = lmdb.open(str(output_dir), map_size=int(1024**4))

        for ep in tqdm(episodes_dir.glob('episode*')):
            episode = int(ep.stem[7:])
            try:
                demo, key_frameids, state_dict_ls, action_ls = get_observation(
                    task_str, variation, episode, rlbench_env
                )
            except (FileNotFoundError, RuntimeError, IndexError) as e:
                print(e)
                return

            gripper_pose = []
            for key_frameid in key_frameids:
                gripper_pose.append({
                    cam: convert_gripper_pose_world_to_image(demo[key_frameid], cam) for cam in args.cameras
                })
                
            outs = {
                'key_frameids': key_frameids,
                'rgb': state_dict_ls['rgb'], # (T, N, H, W, 3)
                'pc': state_dict_ls['pc'], # (T, N, H, W, 3)
                'action': action_ls, # (T, A)
                'gripper_pose': gripper_pose, # [T of dict]
            }

            txn = lmdb_env.begin(write=True)
            txn.put(f'episode{episode}'.encode('ascii'), msgpack.packb(outs))
            txn.commit()

        lmdb_env.close()


if __name__ == "__main__":
    args = Arguments().parse_args()
    generate_keystep_dataset(args)

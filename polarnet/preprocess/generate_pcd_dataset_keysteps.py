from typing import List, Dict, Optional

import os
import argparse
import numpy as np
import collections
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

import open3d as o3d

import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from polarnet.utils.utils import get_assets_dir
from polarnet.config.constants import get_workspace


tasks_use_table_surface = json.load(
    open(f"{get_assets_dir()}/tasks_use_table_surface.json", "r")
)


def process_point_clouds(
    rgb,
    pc,
    gripper_pose=None,
    task=None,
    pc_space="workspace_on_table",
    pc_center_type="gripper",
    pc_radius_norm=True,
    voxel_size=0.01,
    no_rgb=False,
    no_normal=False,
    no_height=False,
):
    rgb = rgb.reshape(-1, 3)
    pc = pc.reshape(-1, 3)
    WORKSPACE = get_workspace(real_robot=False)

    X_BBOX = WORKSPACE["X_BBOX"]
    Y_BBOX = WORKSPACE["Y_BBOX"]
    Z_BBOX = WORKSPACE["Z_BBOX"]
    TABLE_HEIGHT = WORKSPACE["TABLE_HEIGHT"]

    if pc_space in ["workspace", "workspace_on_table"]:
        masks = (
            (pc[:, 0] > X_BBOX[0])
            & (pc[:, 0] < X_BBOX[1])
            & (pc[:, 1] > Y_BBOX[0])
            & (pc[:, 1] < Y_BBOX[1])
            & (pc[:, 2] > Z_BBOX[0])
            & (pc[:, 2] < Z_BBOX[1])
        )
        if pc_space == "workspace_on_table" and task not in tasks_use_table_surface:
            masks = masks & (pc[:, 2] > TABLE_HEIGHT)
            rgb = rgb[masks]
            pc = pc[masks]

    # pcd center and radius
    if pc_center_type == "point":
        pc_center = np.mean(pc, 0)
    elif pc_center_type == "gripper":
        pc_center = gripper_pose[:3]

    if pc_radius_norm:
        pc_radius = np.max(np.sqrt(np.sum((pc - pc_center) ** 2, 1)), keepdims=True)
    else:
        pc_radius = np.ones((1,), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.copy(pc))
    pcd.colors = o3d.utility.Vector3dVector(np.copy(rgb) / 255.0)

    if voxel_size is not None and voxel_size > 0:
        downpcd, _, idxs = pcd.voxel_down_sample_and_trace(
            voxel_size, np.min(pc, 0), np.max(pc, 0)
        )
    else:
        downpcd = pcd

    new_rgb = np.asarray(downpcd.colors) * 2 - 1  # (-1, 1)
    new_pos = np.asarray(downpcd.points)

    # normalized point clouds
    new_ft = (new_pos - pc_center) / pc_radius
    if not no_rgb:
        # use_color
        new_ft = np.concatenate([new_ft, new_rgb], axis=-1)
    if not no_normal:
        # use_normal
        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        new_ft = np.concatenate([new_ft, np.asarray(downpcd.normals)], axis=-1)
    if not no_height:
        # use_height
        heights = np.asarray(downpcd.points)[:, -1]
        heights = heights - TABLE_HEIGHT
        new_ft = np.concatenate([new_ft, heights[:, None]], axis=-1)
    return new_ft, pc_center, pc_radius


def main(args):
    seed = args.seed
    dataset_dir = args.dataset_dir

    if args.seed >= 0:
        keystep_dir = os.path.join(dataset_dir, "keysteps", "seed%d" % seed)
        out_dir = os.path.join(dataset_dir, args.outname, "seed%d" % seed)
    else:
        keystep_dir = os.path.join(dataset_dir, "keysteps")
        out_dir = os.path.join(dataset_dir, args.outname)

    taskvars = os.listdir(keystep_dir)
    taskvars.sort()
    print("#taskvars", len(taskvars))

    os.makedirs(out_dir, exist_ok=True)

    for taskvar in tqdm(taskvars):
        task = taskvar.split("+")[0]

        lmdb_env = lmdb.open(
            os.path.join(keystep_dir, taskvar), readonly=True, lock=False
        )
        lmdb_txn = lmdb_env.begin()

        out_lmdb_env = lmdb.open(
            os.path.join(out_dir, taskvar), map_size=int(1024**4)
        )

        pbar = tqdm(total=lmdb_txn.stat()["entries"])
        for episode_key, value in lmdb_txn.cursor():
            episode_key_dec = episode_key.decode("utf-8")
            if not episode_key_dec.startswith("episode"):
                continue

            value = msgpack.unpackb(value)
            rgbs = value["rgb"][
                :, : args.num_cameras
            ]  # (T, C, H, W, 3) just use the first 3 cameras
            pcs = value["pc"][:, : args.num_cameras]  # (T, C, H, W, 3)
            actions = value["action"]

            outs = collections.defaultdict(list)
            for t, rgb in enumerate(rgbs):
                pcd_ft, pc_center, pc_radius = process_point_clouds(
                    rgbs[t],
                    pcs[t],
                    gripper_pose=actions[t],
                    task=task,
                    pc_space="workspace_on_table",
                    pc_center_type="gripper",
                    pc_radius_norm=True,
                    voxel_size=0.01,
                    no_rgb=args.no_rgb,
                    no_normal=args.no_normal,
                    no_height=args.no_height,
                )
                outs["pc_fts"].append(pcd_ft)
                outs["pc_centers"].append(pc_center)
                outs["pc_radii"].append(pc_radius)
            outs["actions"] = actions

            txn = out_lmdb_env.begin(write=True)
            txn.put(episode_key, msgpack.packb(outs))
            txn.commit()

            pbar.update(1)

        lmdb_env.close()
        out_lmdb_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_cameras", type=int, default=3)
    parser.add_argument("--dataset_dir", type=str, default="data/train_dataset")
    parser.add_argument("--outname", type=str, default="keysteps_pcd")

    parser.add_argument("--no_rgb", action="store_true", default=False)
    parser.add_argument("--no_normal", action="store_true", default=False)
    parser.add_argument("--no_height", action="store_true", default=False)

    args = parser.parse_args()

    main(args)

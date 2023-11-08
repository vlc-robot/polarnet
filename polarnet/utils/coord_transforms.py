from typing import List, Tuple

import numpy as np

import torch
import einops
import json
from scipy.spatial.transform import Rotation as R


def convert_gripper_pose_world_to_image(obs, camera: str) -> Tuple[int, int]:
    '''Convert the gripper pose from world coordinate system to image coordinate system.
    image[v, u] is the gripper location.
    '''
    extrinsics_44 = obs.misc[f"{camera}_camera_extrinsics"].astype(np.float32)
    extrinsics_44 = np.linalg.inv(extrinsics_44)

    intrinsics_33 = obs.misc[f"{camera}_camera_intrinsics"].astype(np.float32)
    intrinsics_34 = np.concatenate([intrinsics_33, np.zeros((3, 1), dtype=np.float32)], 1)

    gripper_pos_31 = obs.gripper_pose[:3].astype(np.float32)[:, None]
    gripper_pos_41 = np.concatenate([gripper_pos_31, np.ones((1, 1), dtype=np.float32)], 0)

    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31[:, 0]

    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


def quaternion_to_discrete_euler(quaternion, resolution: int):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution: int):
    euluer = (discrete_euler * resolution) - 180
    return R.from_euler('xyz', euluer, degrees=True).as_quat()


def euler_to_quat(euler, degrees):
    rotation = R.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = R.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)
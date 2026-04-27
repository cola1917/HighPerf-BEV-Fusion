"""Sensor-frame to ego-frame alignment helpers for lidar points and boxes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pyquaternion import Quaternion


def load_current_lidar_and_boxes_ego(nusc: Any, sample_token: str) -> tuple[np.ndarray, list[object], int]:
    """Load current-frame LIDAR_TOP points/boxes and align both to ego frame.

    Returns:
        lidar_xyz_ego: (N, 3) float32 points in ego frame
        boxes_ego: list of boxes transformed to ego frame
        io_ns: lidar file IO latency in ns
    """
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        return np.empty((0, 3), dtype=np.float32), [], 0

    io_start = time.perf_counter_ns()
    lidar_path = nusc.get_sample_data_path(lidar_token)
    lidar_raw = np.fromfile(lidar_path, dtype=np.float32)
    io_ns = time.perf_counter_ns() - io_start

    if lidar_raw.size == 0 or lidar_raw.size % 5 != 0:
        return np.empty((0, 3), dtype=np.float32), [], io_ns

    lidar_xyz_sensor = lidar_raw.reshape(-1, 5)[:, :3].astype(np.float32, copy=False)

    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])

    lidar_rot = Quaternion(lidar_cs["rotation"]).rotation_matrix.astype(np.float32)
    lidar_trans = np.asarray(lidar_cs["translation"], dtype=np.float32)

    lidar_xyz_ego = (lidar_rot @ lidar_xyz_sensor.T).T + lidar_trans

    _, boxes, _ = nusc.get_sample_data(lidar_token)
    lidar_q = Quaternion(lidar_cs["rotation"])
    for box in boxes:
        box.rotate(lidar_q)
        box.translate(lidar_trans)

    return lidar_xyz_ego, boxes, io_ns

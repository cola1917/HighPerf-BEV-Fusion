"""nuScenes data loading and extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class NuscManager:
    """Manage access to nuScenes mini dataset frame-level camera metadata."""

    CAMERA_CHANNELS = (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )

    def __init__(self, dataroot: str = "/data/nuscenes", version: str = "v1.0-mini") -> None:
        """Initialize nuScenes manager with mini split under the given data root."""
        self.dataroot = str(Path(dataroot))
        self.version = version
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

    def get_frame_data(self, sample_token: str) -> dict[str, dict[str, Any]]:
        """Return 6-camera metadata for a sample token.

        Returns a dict keyed by camera channel. Each entry contains:
        - file_path: Absolute file path of the camera image.
        - camera_intrinsic: 3x3 camera intrinsic matrix as a numpy array.
        - rotation: 3x3 rotation matrix converted from quaternion.
        - translation: 3D translation vector as a numpy array.
        """
        sample = self.nusc.get("sample", sample_token)
        frame_data: dict[str, dict[str, Any]] = {}

        for camera_name in self.CAMERA_CHANNELS:
            sample_data_token = sample["data"].get(camera_name)
            if sample_data_token is None:
                continue

            sample_data = self.nusc.get("sample_data", sample_data_token)
            calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

            rotation_matrix = Quaternion(calibrated_sensor["rotation"]).rotation_matrix

            frame_data[camera_name] = {
                "path": self.nusc.get_sample_data_path(sample_data_token),
                "intrinsic": np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64),
                "rotation": np.asarray(rotation_matrix, dtype=np.float64),
                "translation": np.asarray(calibrated_sensor["translation"], dtype=np.float64),
            }

        return frame_data

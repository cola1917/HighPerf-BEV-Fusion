"""V0 baseline: serial loop pipeline for CAM_FRONT BEV projection."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.core.bev_utils import project_frame_to_bev
from src.core.data_loader import NuscManager


def main() -> None:
    """Main baseline pipeline: load first sample, project CAM_FRONT to BEV, save result."""
    # 1. Initialize nuScenes manager
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot="/data/nuscenes", version="v1.0-mini")
    nusc = nusc_manager.nusc

    # 2. Get first sample
    first_sample = nusc.sample[0]
    sample_token = first_sample["token"]
    print(f"Processing sample token: {sample_token}")

    # 3. Get frame data (all camera extrinsics/intrinsics)
    frame_data = nusc_manager.get_frame_data(sample_token)

    # 4. Extract CAM_FRONT data
    if "CAM_FRONT" not in frame_data:
        raise ValueError("CAM_FRONT not found in frame data")

    cam_front_data = frame_data["CAM_FRONT"]
    cam_front_path = cam_front_data["path"]

    print(f"Loading CAM_FRONT image from: {cam_front_path}")
    image = cv2.imread(cam_front_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {cam_front_path}")

    intrinsic = cam_front_data["intrinsic"]
    rotation = cam_front_data["rotation"]
    translation = cam_front_data["translation"]

    print(f"Image shape: {image.shape}")
    print(f"Intrinsic matrix:\n{intrinsic}")
    print(f"Rotation matrix shape: {rotation.shape}")
    print(f"Translation: {translation}")

    # 5. Project to BEV
    print("Projecting CAM_FRONT to BEV...")
    bev_img = project_frame_to_bev(image, intrinsic, rotation, translation)

    # 6. Save result
    output_path = "/app/output_front_bev.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, bev_img)
    print(f"Saved BEV image to: {output_path}")


if __name__ == "__main__":
    main()

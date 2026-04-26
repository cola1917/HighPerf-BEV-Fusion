"""V0 baseline: serial loop pipeline with all-camera BEV fusion."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.config import CAMERA_NAMES
from src.core.bev_utils import project_frame_to_bev
from src.core.data_loader import NuscManager


def main() -> None:
    """Load first sample, project all cameras to BEV, and save fused output."""
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

    # 4. Loop over all configured cameras and fuse into one total BEV
    total_bev: np.ndarray | None = None

    for camera_name in CAMERA_NAMES:
        if camera_name not in frame_data:
            print(f"Skip {camera_name}: no frame data")
            continue

        cam_data = frame_data[camera_name]
        image_path = cam_data["path"]

        print(f"Loading {camera_name} image from: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skip {camera_name}: failed to load image")
            continue

        bev_img = project_frame_to_bev(
            image=image,
            intrinsic=cam_data["intrinsic"],
            rotation=cam_data["rotation"],
            translation=cam_data["translation"],
        )

        if total_bev is None:
            total_bev = bev_img
        else:
            total_bev = np.maximum(total_bev, bev_img)

    if total_bev is None:
        raise RuntimeError("No valid camera projection generated.")

    # 5. Save fused result
    output_path = "/app/output_total_bev.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, total_bev)
    print(f"Saved fused BEV image to: {output_path}")


if __name__ == "__main__":
    main()

"""V0 baseline: all-camera BEV fusion with LIDAR point cloud overlay."""

from __future__ import annotations

import os

import cv2
import numpy as np

from src.config import (
    BEV_HEIGHT,
    BEV_WIDTH,
    CAMERA_NAMES,
    DATA_ROOT,
    RESOLUTION,
    VERSION,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)
from src.core.bev_utils import project_frame_to_bev
from src.core.data_loader import NuscManager


def _physical_to_bev_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map physical XY coordinates to BEV pixel indices."""
    x_idx = np.round((x - X_MIN) / RESOLUTION).astype(np.int32)
    y_idx = np.round((y - Y_MIN) / RESOLUTION).astype(np.int32)
    y_idx = (BEV_HEIGHT - 1) - y_idx
    return x_idx, y_idx


def _box_category_color(category_name: str) -> tuple[int, int, int]:
    """Category-to-color mapping in BGR format."""
    category = category_name.lower()
    if "vehicle" in category:
        return (255, 0, 0)  # blue
    if "pedestrian" in category:
        return (0, 255, 255)  # yellow
    return (0, 200, 0)  # fallback green


def overlay_bev_box_outlines(
    bev_img: np.ndarray,
    boxes: list[object],
    alpha: float = 0.25,
) -> np.ndarray:
    """Draw BEV boxes with transparent interior and fully visible solid outlines."""
    fill_overlay = np.zeros_like(bev_img)
    outlined = bev_img.copy()

    for box in boxes:
        # Use bottom face corners to draw an oriented rectangle on BEV.
        corners = box.bottom_corners()
        x = corners[0, :]
        y = corners[1, :]
        x_idx, y_idx = _physical_to_bev_pixel(x, y)
        polygon = np.stack([x_idx, y_idx], axis=1).astype(np.int32)

        color = _box_category_color(getattr(box, "name", ""))
        cv2.fillPoly(fill_overlay, [polygon], color=color)
        cv2.polylines(outlined, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(bev_img, 1.0, fill_overlay, alpha, 0.0)
    line_mask = np.any(outlined != bev_img, axis=2)
    blended[line_mask] = outlined[line_mask]
    return blended


def overlay_lidar_on_bev(bev_img: np.ndarray, lidar_points_xyz: np.ndarray) -> np.ndarray:
    """Overlay ego-frame lidar points onto BEV image.

    Uses XY to map points into BEV pixels and colors points by Z height.
    """
    x = lidar_points_xyz[:, 0]
    y = lidar_points_xyz[:, 1]
    z = lidar_points_xyz[:, 2]

    # Keep only points inside configured BEV physical range.
    in_range = (x >= X_MIN) & (x <= X_MAX) & (y >= Y_MIN) & (y <= Y_MAX)
    if not np.any(in_range):
        return bev_img

    x = x[in_range]
    y = y[in_range]
    z = z[in_range]

    # Map physical XY to pixel grid. Y is flipped to keep +Y (front) at the top.
    x_idx = np.round((x - X_MIN) / RESOLUTION).astype(np.int32)
    y_idx = np.round((y - Y_MIN) / RESOLUTION).astype(np.int32)
    y_idx = (BEV_HEIGHT - 1) - y_idx

    valid = (x_idx >= 0) & (x_idx < BEV_WIDTH) & (y_idx >= 0) & (y_idx < BEV_HEIGHT)
    if not np.any(valid):
        return bev_img

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z = z[valid]

    # Height coloring: lower points darker green, higher points brighter yellow-green.
    z_min, z_max = np.percentile(z, [5, 95])
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    point_colors = np.stack(
        [
            np.zeros_like(z_norm),
            (120 + 135 * z_norm).astype(np.uint8),
            (40 + 80 * z_norm).astype(np.uint8),
        ],
        axis=1,
    )

    bev_with_points = bev_img.copy()
    bev_with_points[y_idx, x_idx] = point_colors
    return bev_with_points


def main() -> None:
    """Load first sample, fuse 6-camera BEV, overlay lidar points, and save outputs."""
    # 1. Initialize nuScenes manager
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot=DATA_ROOT, version=VERSION)
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

    # 5. Read LIDAR_TOP point cloud (.bin) and overlay to BEV
    lidar_token = first_sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        raise RuntimeError("LIDAR_TOP not found in sample data.")

    lidar_path, lidar_boxes, _ = nusc.get_sample_data(lidar_token)
    print(f"Loading LIDAR_TOP points from: {lidar_path}")
    print(f"Loaded lidar boxes: {len(lidar_boxes)}")

    # nuScenes lidar binary is float32 with 5 values per point: x, y, z, intensity, ring index.
    lidar_raw = np.fromfile(lidar_path, dtype=np.float32)
    if lidar_raw.size == 0 or lidar_raw.size % 5 != 0:
        raise RuntimeError(f"Unexpected lidar data format in file: {lidar_path}")

    lidar_points = lidar_raw.reshape(-1, 5)
    lidar_xyz = lidar_points[:, :3]
    print(f"Loaded lidar points: {lidar_xyz.shape[0]}")

    total_bev_with_lidar = overlay_lidar_on_bev(total_bev, lidar_xyz)
    total_bev_with_lidar = overlay_bev_box_outlines(total_bev_with_lidar, lidar_boxes, alpha=0.45)

    output_final_path = "/app/output_total_bev_lidar_box.jpg"
    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
    cv2.imwrite(output_final_path, total_bev_with_lidar)
    print(f"Saved final BEV (with LIDAR + BOX) to: {output_final_path}")

    print("Visual checks:")
    print("1) Curb/road edges should align with dense green lidar traces.")
    print("2) Building wall regions should look like line-shaped point clusters in BEV.")


if __name__ == "__main__":
    main()

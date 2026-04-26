"""V1/V2 engine: asyncio orchestration + ProcessPool parallel projection."""

from __future__ import annotations

import asyncio
import gc
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import cv2
import numpy as np

from src.config import BEV_HEIGHT, BEV_WIDTH, CAMERA_NAMES, DATA_ROOT, RESOLUTION, VERSION, X_MAX, X_MIN, Y_MAX, Y_MIN
from src.core.bev_utils import project_frame_to_bev
from src.core.data_loader import NuscManager


def _physical_to_bev_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map ego-frame XY coordinates to BEV pixel indices."""
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
    """Render lidar points on BEV using Z-height JET colormap."""
    x = lidar_points_xyz[:, 0]
    y = lidar_points_xyz[:, 1]
    z = lidar_points_xyz[:, 2]

    in_range = (x >= X_MIN) & (x <= X_MAX) & (y >= Y_MIN) & (y <= Y_MAX)
    if not np.any(in_range):
        return bev_img

    x = x[in_range]
    y = y[in_range]
    z = z[in_range]

    x_idx, y_idx = _physical_to_bev_pixel(x, y)
    valid = (x_idx >= 0) & (x_idx < BEV_WIDTH) & (y_idx >= 0) & (y_idx < BEV_HEIGHT)
    if not np.any(valid):
        return bev_img

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z = z[valid]

    z_norm = np.clip((z - float(z.min())) / max(float(z.max()) - float(z.min()), 1e-6), 0.0, 1.0)
    z_uint8 = (z_norm * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(z_uint8.reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]

    bev_out = bev_img.copy()
    bev_out[y_idx, x_idx] = colors
    return bev_out


def _project_single_camera_worker(
    camera_name: str,
    image_path: str,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[str, np.ndarray] | None:
    """Worker process: load one camera image and project it to BEV."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    bev = project_frame_to_bev(
        image=image,
        intrinsic=intrinsic,
        rotation=rotation,
        translation=translation,
    )

    # Release per-camera temporary memory in worker process ASAP.
    del image
    gc.collect()
    return camera_name, bev


def _prepare_camera_job(camera_name: str, frame_data: dict[str, dict[str, Any]]) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray] | None:
    """Prepare one camera payload and validate required inputs."""
    cam_data = frame_data.get(camera_name)
    if cam_data is None:
        return None

    image_path = cam_data.get("path")
    if not image_path or not os.path.exists(image_path):
        return None

    return (
        camera_name,
        str(image_path),
        np.asarray(cam_data["intrinsic"], dtype=np.float64),
        np.asarray(cam_data["rotation"], dtype=np.float64),
        np.asarray(cam_data["translation"], dtype=np.float64),
    )


async def _collect_camera_jobs(frame_data: dict[str, dict[str, Any]]) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """Asynchronously prepare camera jobs for process pool execution."""
    tasks = [
        asyncio.to_thread(_prepare_camera_job, camera_name, frame_data)
        for camera_name in CAMERA_NAMES
    ]
    prepared = await asyncio.gather(*tasks)
    return [item for item in prepared if item is not None]


async def run_pipeline() -> None:
    """Run concurrent BEV projection pipeline for one sample."""
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot=DATA_ROOT, version=VERSION)
    nusc = nusc_manager.nusc

    first_sample = nusc.sample[0]
    sample_token = first_sample["token"]
    print(f"Processing sample token: {sample_token}")

    frame_data = nusc_manager.get_frame_data(sample_token)
    camera_jobs = await _collect_camera_jobs(frame_data)

    if not camera_jobs:
        raise RuntimeError("No valid camera jobs available for concurrent processing.")

    max_workers = min(len(camera_jobs), os.cpu_count() or 1)
    print(f"Launching ProcessPoolExecutor with {max_workers} workers")

    total_bev: np.ndarray | None = None
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            loop.run_in_executor(executor, _project_single_camera_worker, *job)
            for job in camera_jobs
        ]

        for fut in asyncio.as_completed(futures):
            result = await fut
            if result is None:
                continue

            camera_name, bev_img = result
            print(f"Projected {camera_name}")

            if total_bev is None:
                total_bev = bev_img
            else:
                total_bev = np.maximum(total_bev, bev_img)

            # Release finished task buffers immediately to reduce peak memory.
            del bev_img
            gc.collect()

    if total_bev is None:
        raise RuntimeError("Concurrent projection failed: no BEV outputs produced.")

    # Overlay LIDAR_TOP on the fused BEV.
    lidar_token = first_sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        raise RuntimeError("LIDAR_TOP not found in sample data.")

    lidar_path, lidar_boxes, _ = nusc.get_sample_data(lidar_token)
    lidar_raw = np.fromfile(lidar_path, dtype=np.float32)
    if lidar_raw.size == 0 or lidar_raw.size % 5 != 0:
        raise RuntimeError(f"Unexpected lidar data format in file: {lidar_path}")

    lidar_xyz = lidar_raw.reshape(-1, 5)[:, :3]
    total_bev_lidar = overlay_lidar_on_bev(total_bev, lidar_xyz)
    total_bev_lidar = overlay_bev_box_outlines(total_bev_lidar, lidar_boxes, alpha=0.25)

    output_final_path = "/app/output_total_bev_concurrent_lidar_box.jpg"
    os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
    cv2.imwrite(output_final_path, total_bev_lidar)
    print(f"Saved final concurrent BEV (with LIDAR + BOX) to: {output_final_path}")

    del lidar_raw, lidar_xyz, lidar_boxes, total_bev_lidar
    gc.collect()

    # Final cleanup for container memory friendliness.
    del total_bev
    gc.collect()


def main() -> None:
    """Entry point for the concurrent engine."""
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()

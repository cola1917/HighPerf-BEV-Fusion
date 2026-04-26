"""V3 engine: LUT-accelerated production pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

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
from src.core.bev_utils import build_bev_remap_lut, fast_remap_kernel
from src.core.data_loader import NuscManager


LUT_CACHE_DIR = Path("/app/cache/ultimate_luts")
LUT_INDEX_PATH = LUT_CACHE_DIR / "index.json"
OUTPUT_PATH = "/app/output_total_bev_ultimate_lidar_box.jpg"


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
        return (255, 0, 0)
    if "pedestrian" in category:
        return (0, 255, 255)
    return (0, 200, 0)


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
    """Overlay ego-frame lidar points onto BEV image using JET height coloring."""
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


def _camera_lut_hash(camera_name: str, cam_data: dict[str, Any]) -> str:
    """Build a stable hash key for reusable LUT caching across samples."""
    hasher = hashlib.sha1()
    hasher.update(camera_name.encode("utf-8"))
    hasher.update(np.asarray(cam_data["intrinsic"], dtype=np.float64).tobytes())
    hasher.update(np.asarray(cam_data["rotation"], dtype=np.float64).tobytes())
    hasher.update(np.asarray(cam_data["translation"], dtype=np.float64).tobytes())
    hasher.update(np.asarray([BEV_HEIGHT, BEV_WIDTH, RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX], dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _load_lut_index() -> dict[str, str]:
    if not LUT_INDEX_PATH.exists():
        return {}
    with LUT_INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_lut_index(index: dict[str, str]) -> None:
    LUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with LUT_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _load_or_build_luts(
    frame_data: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int], float]:
    """Load per-camera LUTs from reusable file cache or build and persist them."""
    start = time.perf_counter()
    LUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_lut_index()

    luts: dict[str, dict[str, np.ndarray]] = {}
    stats = {"hit": 0, "miss": 0}

    for camera_name in CAMERA_NAMES:
        cam_data = frame_data.get(camera_name)
        if cam_data is None:
            continue

        lut_hash = _camera_lut_hash(camera_name, cam_data)
        cache_filename = f"lut_{camera_name}_{lut_hash}.npz"
        cache_path = LUT_CACHE_DIR / cache_filename
        index_key = f"{camera_name}:{lut_hash}"

        if index.get(index_key) and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            luts[camera_name] = {
                "u_map": cached["u_map"],
                "v_map": cached["v_map"],
                "mask": cached["mask"],
            }
            cached.close()
            stats["hit"] += 1
            continue

        u_map, v_map, mask = build_bev_remap_lut(
            intrinsic=np.asarray(cam_data["intrinsic"], dtype=np.float64),
            rotation=np.asarray(cam_data["rotation"], dtype=np.float64),
            translation=np.asarray(cam_data["translation"], dtype=np.float64),
        )
        luts[camera_name] = {"u_map": u_map, "v_map": v_map, "mask": mask}
        np.savez_compressed(cache_path, u_map=u_map, v_map=v_map, mask=mask)
        index[index_key] = cache_filename
        stats["miss"] += 1

    _save_lut_index(index)
    return luts, stats, time.perf_counter() - start


def _worker_remap_to_shared(
    camera_name: str,
    image_path: str,
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask: np.ndarray,
    camera_index: int,
    shm_name: str,
) -> str | None:
    """Worker: read image and write projected BEV slice into shared memory."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        shared_view = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        bev_slice = shared_view[camera_index]
        bev_slice.fill(0)

        fast_remap_kernel(image, u_map, v_map, mask, bev_slice)
        del bev_slice, shared_view, image
        gc.collect()
        return camera_name
    finally:
        shm.close()


def main() -> None:
    """Entry point for the ultimate engine."""
    total_start = time.perf_counter()
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot=DATA_ROOT, version=VERSION)
    nusc = nusc_manager.nusc

    first_sample = nusc.sample[0]
    sample_token = first_sample["token"]
    print(f"Processing sample token: {sample_token}")

    frame_data = nusc_manager.get_frame_data(sample_token)

    lut_start = time.perf_counter()
    luts, lut_stats, lut_elapsed = _load_or_build_luts(frame_data)
    print(f"LUT cache hits={lut_stats['hit']}, misses={lut_stats['miss']}, time={lut_elapsed:.3f}s")
    print(f"LUT phase elapsed={time.perf_counter() - lut_start:.3f}s")

    jobs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for camera_index, camera_name in enumerate(CAMERA_NAMES):
        cam_data = frame_data.get(camera_name)
        lut_data = luts.get(camera_name)
        if cam_data is None or lut_data is None:
            continue

        image_path = cam_data.get("path")
        if not image_path or not os.path.exists(image_path):
            continue

        jobs.append(
            (
                camera_name,
                str(image_path),
                lut_data["u_map"],
                lut_data["v_map"],
                lut_data["mask"],
                camera_index,
            )
        )

    if not jobs:
        raise RuntimeError("No valid camera jobs available for ultimate engine.")

    shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    try:
        shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        shared_array.fill(0)

        max_workers = min(len(jobs), os.cpu_count() or 1)
        print(f"Launching ProcessPoolExecutor with {max_workers} workers")

        proj_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_remap_to_shared, *job, shm.name) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    print("Skipped one camera due to read failure")
                else:
                    print(f"Projected {result}")
        proj_elapsed = time.perf_counter() - proj_start

        fuse_start = time.perf_counter()
        total_bev = np.max(shared_array, axis=0)
        total_bev = cv2.flip(total_bev, 0)
        fuse_elapsed = time.perf_counter() - fuse_start

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

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        cv2.imwrite(OUTPUT_PATH, total_bev_lidar)
        print(f"Saved ultimate fused BEV image to: {OUTPUT_PATH}")

        print(
            "Timing summary: "
            f"prepare/lut={lut_elapsed:.3f}s, "
            f"projection={proj_elapsed:.3f}s, "
            f"fusion={fuse_elapsed:.3f}s, "
            f"total={time.perf_counter() - total_start:.3f}s"
        )

        del lidar_raw, lidar_xyz, lidar_boxes, total_bev_lidar, total_bev, shared_array
        gc.collect()
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()

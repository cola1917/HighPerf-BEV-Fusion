"""Production BEV engine: TurboJPEG IO + shared memory remap + temporal multimodal fusion."""

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
import numba
import numpy as np

from src.config import BEV_HEIGHT, BEV_WIDTH, CAMERA_NAMES, DATA_ROOT, RESOLUTION, VERSION, X_MAX, X_MIN, Y_MAX, Y_MIN
from src.core.bev_utils import build_bev_remap_lut, fast_remap_kernel
from src.core.data_loader import NuscManager
from src.core.sensor_alignment import load_current_lidar_and_boxes_ego

try:
    from turbojpeg import TJPF_BGR, TurboJPEG
except Exception:  # pragma: no cover - runtime fallback for missing dependency
    TurboJPEG = None
    TJPF_BGR = None


LUT_CACHE_DIR = Path("/app/cache/production_luts")
LUT_INDEX_PATH = LUT_CACHE_DIR / "index.json"
OUTPUT_PATH = "/app/output_total_bev_production.jpg"


_DECODER = None


def _get_decoder() -> Any:
    global _DECODER
    if _DECODER is None and TurboJPEG is not None:
        _DECODER = TurboJPEG()
    return _DECODER


def _decode_image_fast(image_path: str) -> np.ndarray | None:
    """Decode JPEG with TurboJPEG, fallback to cv2.imdecode if unavailable."""
    with open(image_path, "rb") as f:
        encoded = f.read()

    decoder = _get_decoder()
    if decoder is not None:
        return decoder.decode(encoded, pixel_format=TJPF_BGR)

    encoded_np = np.frombuffer(encoded, dtype=np.uint8)
    return cv2.imdecode(encoded_np, cv2.IMREAD_COLOR)


def _camera_lut_hash(camera_name: str, cam_data: dict[str, Any]) -> str:
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


def _load_or_build_luts(frame_data: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int], int]:
    """Load or build LUT for each camera once, cached on disk for startup speed."""
    LUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_lut_index()
    stats = {"hit": 0, "miss": 0}
    elapsed_ns = 0
    luts: dict[str, dict[str, np.ndarray]] = {}

    for camera_name in CAMERA_NAMES:
        cam_data = frame_data.get(camera_name)
        if cam_data is None:
            continue

        lut_hash = _camera_lut_hash(camera_name, cam_data)
        cache_name = f"lut_{camera_name}_{lut_hash}.npz"
        cache_path = LUT_CACHE_DIR / cache_name
        index_key = f"{camera_name}:{lut_hash}"

        if index.get(index_key) and cache_path.exists():
            t0 = time.perf_counter_ns()
            cached = np.load(cache_path, allow_pickle=False)
            luts[camera_name] = {
                "u_map": cached["u_map"],
                "v_map": cached["v_map"],
                "mask": cached["mask"],
            }
            cached.close()
            elapsed_ns += time.perf_counter_ns() - t0
            stats["hit"] += 1
            continue

        t0 = time.perf_counter_ns()
        u_map, v_map, mask = build_bev_remap_lut(
            intrinsic=np.asarray(cam_data["intrinsic"], dtype=np.float64),
            rotation=np.asarray(cam_data["rotation"], dtype=np.float64),
            translation=np.asarray(cam_data["translation"], dtype=np.float64),
        )
        luts[camera_name] = {"u_map": u_map, "v_map": v_map, "mask": mask}
        np.savez_compressed(cache_path, u_map=u_map, v_map=v_map, mask=mask)
        index[index_key] = cache_name
        elapsed_ns += time.perf_counter_ns() - t0
        stats["miss"] += 1

    _save_lut_index(index)
    return luts, stats, elapsed_ns


def _worker_remap_to_shared_turbo(
    camera_name: str,
    image_path: str,
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask: np.ndarray,
    camera_index: int,
    shm_name: str,
) -> tuple[str, int, int] | None:
    io_start = time.perf_counter_ns()
    image = _decode_image_fast(image_path)
    io_ns = time.perf_counter_ns() - io_start
    if image is None:
        return None

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        bev_slice = shared_array[camera_index]
        bev_slice.fill(0)

        compute_start = time.perf_counter_ns()
        fast_remap_kernel(image, u_map, v_map, mask, bev_slice)
        compute_ns = time.perf_counter_ns() - compute_start

        del bev_slice, shared_array, image
        gc.collect()
        return camera_name, io_ns, compute_ns
    finally:
        shm.close()


def _prepare_current_lidar_and_boxes(nusc: Any, sample_token: str) -> tuple[np.ndarray, list[object], int]:
    """Load current-frame lidar and boxes only (same logic as previous engines)."""
    return load_current_lidar_and_boxes_ego(nusc, sample_token)


def _box_category_color(category_name: str) -> tuple[int, int, int]:
    category = category_name.lower()
    if "vehicle" in category:
        return (255, 0, 0)
    if "pedestrian" in category:
        return (0, 255, 255)
    return (0, 200, 0)


def _physical_to_bev_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.round((x - X_MIN) / RESOLUTION).astype(np.int32)
    y_idx = np.round((y - Y_MIN) / RESOLUTION).astype(np.int32)
    y_idx = (BEV_HEIGHT - 1) - y_idx
    return x_idx, y_idx


@numba.njit(parallel=True)
def _overlay_points_kernel(
    bev_img: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    b: np.uint8,
    g: np.uint8,
    r: np.uint8,
) -> None:
    """Numba-accelerated multi-threaded pixel overlay with atomic max semantics.
    
    Uses max() instead of if-then-assign to avoid race conditions when multiple
    parallel threads write to the same BEV pixel. This ensures semantic correctness
    even under thread contention: the final value will be the element-wise maximum.
    """
    h, w, _ = bev_img.shape
    n = x_idx.shape[0]
    for i in numba.prange(n):
        x = x_idx[i]
        y = y_idx[i]
        if 0 <= x < w and 0 <= y < h:
            # Use max() for atomic-like semantics: multiple concurrent writes
            # to the same pixel will converge to element-wise maximum.
            # This is safe even under numba.prange parallelism.
            bev_img[y, x, 0] = max(bev_img[y, x, 0], b)
            bev_img[y, x, 1] = max(bev_img[y, x, 1], g)
            bev_img[y, x, 2] = max(bev_img[y, x, 2], r)


def _overlay_points_numba(bev_img: np.ndarray, points_xyz: np.ndarray, color: tuple[int, int, int]) -> None:
    if points_xyz.size == 0:
        return

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    in_range = (x >= X_MIN) & (x <= X_MAX) & (y >= Y_MIN) & (y <= Y_MAX)
    if not np.any(in_range):
        return

    x_idx, y_idx = _physical_to_bev_pixel(x[in_range], y[in_range])
    _overlay_points_kernel(
        bev_img,
        x_idx.astype(np.int32),
        y_idx.astype(np.int32),
        np.uint8(color[0]),
        np.uint8(color[1]),
        np.uint8(color[2]),
    )


def _overlay_boxes_on_bev(
    bev_img: np.ndarray,
    boxes: list[object],
    alpha: float = 0.25,
) -> np.ndarray:
    fill_overlay = np.zeros_like(bev_img)
    outlined = bev_img.copy()

    for box in boxes:
        corners = box.bottom_corners()
        x_idx, y_idx = _physical_to_bev_pixel(corners[0, :], corners[1, :])
        polygon = np.stack([x_idx, y_idx], axis=1).astype(np.int32)

        color = _box_category_color(getattr(box, "name", ""))
        cv2.fillPoly(fill_overlay, [polygon], color=color)
        cv2.polylines(outlined, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(bev_img, 1.0, fill_overlay, alpha, 0.0)
    line_mask = np.any(outlined != bev_img, axis=2)
    blended[line_mask] = outlined[line_mask]
    return blended


def _run_frame(
    nusc_manager: NuscManager,
    nusc: Any,
    sample_token: str,
    luts: dict[str, dict[str, np.ndarray]],
    shared_array: np.ndarray,
    shm_name: str,
    executor: ProcessPoolExecutor,
) -> tuple[np.ndarray | None, int]:
    frame_data = nusc_manager.get_frame_data(sample_token)

    io_ns = 0
    compute_ns = 0
    shared_array.fill(0)

    futures = []
    for camera_index, camera_name in enumerate(CAMERA_NAMES):
        cam_data = frame_data.get(camera_name)
        lut_data = luts.get(camera_name)
        if cam_data is None or lut_data is None:
            continue
        image_path = cam_data.get("path")
        if not image_path or not os.path.exists(image_path):
            continue

        futures.append(
            executor.submit(
                _worker_remap_to_shared_turbo,
                camera_name,
                image_path,
                lut_data["u_map"],
                lut_data["v_map"],
                lut_data["mask"],
                camera_index,
                shm_name,
            )
        )

    for future in as_completed(futures):
        result = future.result()
        if result is None:
            continue
        _, io_cost, compute_cost = result
        io_ns += io_cost
        compute_ns += compute_cost

    fuse_start = time.perf_counter_ns()
    total_bev = np.max(shared_array, axis=0)
    total_bev = cv2.flip(total_bev, 0)
    fuse_ns = time.perf_counter_ns() - fuse_start
    compute_ns += fuse_ns

    multimodal_start = time.perf_counter_ns()
    cur_lidar, boxes, lidar_io_ns = _prepare_current_lidar_and_boxes(nusc, sample_token)
    io_ns += lidar_io_ns

    _overlay_points_numba(total_bev, cur_lidar, (0, 255, 120))
    total_bev = _overlay_boxes_on_bev(total_bev, boxes, alpha=0.22)
    multimodal_ns = time.perf_counter_ns() - multimodal_start
    compute_ns += multimodal_ns

    return total_bev, len(cur_lidar)


def main() -> None:
    total_start = time.perf_counter_ns()
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot=DATA_ROOT, version=VERSION)
    nusc = nusc_manager.nusc

    first_sample = nusc.sample[0]
    target_sample_token = first_sample["token"]

    init_frame_data = nusc_manager.get_frame_data(first_sample["token"])
    luts, lut_stats, lut_elapsed_ns = _load_or_build_luts(init_frame_data)
    print(f"LUT cache hits={lut_stats['hit']} misses={lut_stats['miss']} elapsed={lut_elapsed_ns / 1e6:.3f} ms")
    print(f"Rendering sample token: {target_sample_token}")

    shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)

    try:
        with ProcessPoolExecutor(max_workers=min(len(CAMERA_NAMES), os.cpu_count() or 1)) as executor:
            image, cur_n = _run_frame(nusc_manager, nusc, target_sample_token, luts, shared_array, shm.name, executor)
            if image is None:
                raise RuntimeError("Failed to render production frame.")

            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            cv2.imwrite(OUTPUT_PATH, image)
            print(f"Saved production BEV image to: {OUTPUT_PATH}")
            print(f"Current lidar points: {cur_n}")

        print(f"Total wall time: {(time.perf_counter_ns() - total_start) / 1e6:.3f} ms")
    finally:
        del shared_array
        shm.close()
        shm.unlink()
        gc.collect()


if __name__ == "__main__":
    main()

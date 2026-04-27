"""Performance benchmark comparing baseline, concurrent, ultimate, and production engines."""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Callable

import cv2
import numpy as np

from src.baseline import overlay_bev_box_outlines, overlay_lidar_on_bev
from src.config import BEV_HEIGHT, BEV_WIDTH, CAMERA_NAMES, DATA_ROOT, RESOLUTION, VERSION, X_MAX, X_MIN, Y_MAX, Y_MIN
from src.core.bev_utils import build_bev_remap_lut, fast_remap_kernel, project_frame_to_bev
from src.core.data_loader import NuscManager
from src.core.sensor_alignment import load_current_lidar_and_boxes_ego
from src.production_engine import (
    _load_or_build_luts as _load_or_build_luts_production,
    _overlay_boxes_on_bev,
    _overlay_points_numba,
    _prepare_current_lidar_and_boxes,
    _worker_remap_to_shared_turbo,
)
from src.ultimate_engine import _load_or_build_luts, _worker_remap_to_shared


WARMUP_FRAMES = 5
BENCHMARK_FRAMES = 100
MAX_WORKERS = min(len(CAMERA_NAMES), os.cpu_count() or 1)


@dataclass
class FrameStat:
    frame_index: int
    latency_ns: int
    io_ns: int
    compute_ns: int
    prep_ns: int
    fuse_ns: int


@dataclass
class ModeResult:
    mode: str
    frame_stats: list[FrameStat]
    cache_hits: int = 0
    cache_misses: int = 0
    lut_elapsed_ns: int = 0


@dataclass
class CameraJob:
    camera_name: str
    image_path: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    camera_index: int


@dataclass
class FrameJobs:
    sample_token: str
    camera_jobs: list[CameraJob]


@dataclass
class FrameOverlayData:
    lidar_xyz: np.ndarray | None
    lidar_boxes: list[object]
    io_ns: int


_CACHE_DIR = "/app/cache/ultimate_luts"


def _physical_to_bev_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.round((x - X_MIN) / RESOLUTION).astype(np.int32)
    y_idx = np.round((y - Y_MIN) / RESOLUTION).astype(np.int32)
    y_idx = (BEV_HEIGHT - 1) - y_idx
    return x_idx, y_idx


def _collect_scene_sample_tokens(nusc: Any, start_sample_token: str, max_frames: int) -> list[str]:
    sample_tokens = [start_sample_token]
    current = nusc.get("sample", start_sample_token)
    while current["next"] and len(sample_tokens) < max_frames:
        sample_tokens.append(current["next"])
        current = nusc.get("sample", current["next"])
    return sample_tokens


def _benchmark_ultimate_worker(
    camera_name: str,
    image_path: str,
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask: np.ndarray,
    camera_index: int,
    shm_name: str,
) -> tuple[str, int, int] | None:
    io_start = time.perf_counter_ns()
    image = cv2.imread(image_path)
    io_ns = time.perf_counter_ns() - io_start
    if image is None:
        return None

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        shared_view = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        bev_slice = shared_view[camera_index]
        bev_slice.fill(0)

        compute_start = time.perf_counter_ns()
        fast_remap_kernel(image, u_map, v_map, mask, bev_slice)
        compute_ns = time.perf_counter_ns() - compute_start

        del bev_slice, shared_view, image
        gc.collect()
        return camera_name, io_ns, compute_ns
    finally:
        shm.close()


def _load_frame_jobs(nusc_manager: NuscManager, sample_token: str) -> FrameJobs:
    frame_data = nusc_manager.get_frame_data(sample_token)
    camera_jobs: list[CameraJob] = []

    for camera_index, camera_name in enumerate(CAMERA_NAMES):
        cam_data = frame_data.get(camera_name)
        if cam_data is None:
            continue

        image_path = cam_data.get("path")
        if not image_path or not os.path.exists(image_path):
            continue

        camera_jobs.append(
            CameraJob(
                camera_name=camera_name,
                image_path=str(image_path),
                intrinsic=np.asarray(cam_data["intrinsic"], dtype=np.float64),
                rotation=np.asarray(cam_data["rotation"], dtype=np.float64),
                translation=np.asarray(cam_data["translation"], dtype=np.float64),
                camera_index=camera_index,
            )
        )

    return FrameJobs(sample_token=sample_token, camera_jobs=camera_jobs)


def _load_overlay_data(nusc: Any, sample_token: str) -> FrameOverlayData:
    lidar_xyz, lidar_boxes, io_ns = load_current_lidar_and_boxes_ego(nusc, sample_token)
    if lidar_xyz.size == 0:
        return FrameOverlayData(lidar_xyz=None, lidar_boxes=lidar_boxes, io_ns=io_ns)
    return FrameOverlayData(lidar_xyz=lidar_xyz, lidar_boxes=lidar_boxes, io_ns=io_ns)


def _finalize_bev(total_bev: np.ndarray, overlay_data: FrameOverlayData) -> np.ndarray:
    total_bev = cv2.flip(total_bev, 0)
    if overlay_data.lidar_xyz is not None:
        total_bev = overlay_lidar_on_bev(total_bev, overlay_data.lidar_xyz)
        total_bev = overlay_bev_box_outlines(total_bev, overlay_data.lidar_boxes, alpha=0.25)
    return total_bev


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _print_mode_summary(result: ModeResult) -> None:
    latencies_ms = [s.latency_ns / 1e6 for s in result.frame_stats]
    io_ms = [s.io_ns / 1e6 for s in result.frame_stats]
    compute_ms = [s.compute_ns / 1e6 for s in result.frame_stats]

    avg_latency = float(statistics.mean(latencies_ms)) if latencies_ms else 0.0
    p99_latency = _percentile(latencies_ms, 99.0)
    total_time_s = sum(s.latency_ns for s in result.frame_stats) / 1e9
    throughput = (len(result.frame_stats) / total_time_s) if total_time_s > 0 else 0.0

    io_total_ms = sum(io_ms)
    compute_total_ms = sum(compute_ms)
    total_stage_ms = io_total_ms + compute_total_ms
    io_ratio = (io_total_ms / total_stage_ms * 100.0) if total_stage_ms > 0 else 0.0
    compute_ratio = (compute_total_ms / total_stage_ms * 100.0) if total_stage_ms > 0 else 0.0

    print("\n============================================================")
    print(f"Mode: {result.mode}")
    print("------------------------------------------------------------")
    print(f"Frames Processed : {len(result.frame_stats)}")
    print(f"Average Latency  : {avg_latency:.3f} ms")
    print(f"P99 Latency      : {p99_latency:.3f} ms")
    print(f"Throughput       : {throughput:.3f} FPS")
    print(f"IO Total         : {io_total_ms:.3f} ms")
    print(f"Compute Total    : {compute_total_ms:.3f} ms")
    print(f"IO Share         : {io_ratio:.2f}%")
    print(f"Compute Share    : {compute_ratio:.2f}%")
    if result.mode in ("ultimate", "production"):
        print(f"LUT Cache Hits   : {result.cache_hits}")
        print(f"LUT Cache Misses : {result.cache_misses}")
        print(f"LUT Elapsed      : {result.lut_elapsed_ns / 1e6:.3f} ms")
    print("------------------------------------------------------------")
    print("IO vs Compute chart data:")
    print(json.dumps({"labels": ["IO", "Compute"], "values": [round(io_ratio, 2), round(compute_ratio, 2)]}, indent=2))
    print("============================================================")


def _warmup_luts_and_numba(nusc_manager: NuscManager, nusc: Any, sample_tokens: list[str]) -> None:
    """Warm up 5 frames so Numba compiles and LUT cache is populated."""
    for sample_token in sample_tokens:
        frame_jobs = _load_frame_jobs(nusc_manager, sample_token)
        for job in frame_jobs.camera_jobs:
            image = cv2.imread(job.image_path)
            if image is None:
                continue
            bev = project_frame_to_bev(
                image=image,
                intrinsic=job.intrinsic,
                rotation=job.rotation,
                translation=job.translation,
            )
            del image, bev
            gc.collect()
        overlay_data = _load_overlay_data(nusc, sample_token)
        del overlay_data
        gc.collect()


def _warmup_concurrent(nusc_manager: NuscManager, nusc: Any, sample_tokens: list[str]) -> None:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for sample_token in sample_tokens:
            frame_jobs = _load_frame_jobs(nusc_manager, sample_token)
            jobs = frame_jobs.camera_jobs
            futures = []
            for job in jobs:
                futures.append(
                    executor.submit(
                        _baseline_worker,
                        job.camera_name,
                        job.image_path,
                        job.intrinsic,
                        job.rotation,
                        job.translation,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    _, bev_img, _, _ = result
                    del bev_img
            overlay_data = _load_overlay_data(nusc, sample_token)
            del overlay_data
            gc.collect()


def _warmup_ultimate(nusc_manager: NuscManager, nusc: Any, sample_tokens: list[str]) -> None:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for sample_token in sample_tokens:
            frame_jobs = _load_frame_jobs(nusc_manager, sample_token)
            luts, _, _ = _load_or_build_luts(
                {
                    job.camera_name: {
                        "path": job.image_path,
                        "intrinsic": job.intrinsic,
                        "rotation": job.rotation,
                        "translation": job.translation,
                    }
                    for job in frame_jobs.camera_jobs
                }
            )

            shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
            shm = shared_memory.SharedMemory(create=True, size=shm_size)
            try:
                shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
                shared_array.fill(0)
                futures = []
                for job in frame_jobs.camera_jobs:
                    lut_data = luts.get(job.camera_name)
                    if lut_data is None:
                        continue
                    futures.append(
                        executor.submit(
                            _worker_remap_to_shared,
                            job.camera_name,
                            job.image_path,
                            lut_data["u_map"],
                            lut_data["v_map"],
                            lut_data["mask"],
                            job.camera_index,
                            shm.name,
                        )
                    )
                for future in as_completed(futures):
                    future.result()
                del shared_array
            finally:
                shm.close()
                shm.unlink()
            overlay_data = _load_overlay_data(nusc, sample_token)
            del overlay_data
            gc.collect()


def _warmup_production(nusc_manager: NuscManager, nusc: Any, sample_tokens: list[str]) -> None:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for sample_token in sample_tokens:
            frame_jobs = _load_frame_jobs(nusc_manager, sample_token)
            luts, _, _ = _load_or_build_luts_production(
                {
                    job.camera_name: {
                        "path": job.image_path,
                        "intrinsic": job.intrinsic,
                        "rotation": job.rotation,
                        "translation": job.translation,
                    }
                    for job in frame_jobs.camera_jobs
                }
            )

            shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
            shm = shared_memory.SharedMemory(create=True, size=shm_size)
            try:
                shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
                shared_array.fill(0)
                futures = []
                for job in frame_jobs.camera_jobs:
                    lut_data = luts.get(job.camera_name)
                    if lut_data is None:
                        continue
                    futures.append(
                        executor.submit(
                            _worker_remap_to_shared_turbo,
                            job.camera_name,
                            job.image_path,
                            lut_data["u_map"],
                            lut_data["v_map"],
                            lut_data["mask"],
                            job.camera_index,
                            shm.name,
                        )
                    )
                for future in as_completed(futures):
                    future.result()
                del shared_array
            finally:
                shm.close()
                shm.unlink()

            _prepare_current_lidar_and_boxes(nusc, sample_token)
            gc.collect()


def _baseline_worker(
    camera_name: str,
    image_path: str,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[str, np.ndarray, int, int] | None:
    io_start = time.perf_counter_ns()
    image = cv2.imread(image_path)
    io_ns = time.perf_counter_ns() - io_start
    if image is None:
        return None
    compute_start = time.perf_counter_ns()
    bev_img = project_frame_to_bev(
        image=image,
        intrinsic=intrinsic,
        rotation=rotation,
        translation=translation,
    )
    compute_ns = time.perf_counter_ns() - compute_start
    del image
    gc.collect()
    return camera_name, bev_img, io_ns, compute_ns


def _concurrent_worker(
    camera_name: str,
    image_path: str,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[str, np.ndarray, int, int] | None:
    return _baseline_worker(camera_name, image_path, intrinsic, rotation, translation)


def _run_baseline_frame(nusc: Any, frame_jobs: FrameJobs) -> tuple[FrameStat, np.ndarray | None]:
    frame_start = time.perf_counter_ns()
    total_bev: np.ndarray | None = None
    io_ns = 0
    compute_ns = 0

    for job in frame_jobs.camera_jobs:
        result = _baseline_worker(job.camera_name, job.image_path, job.intrinsic, job.rotation, job.translation)
        if result is None:
            continue
        _, bev_img, io_cost, compute_cost = result
        io_ns += io_cost
        compute_ns += compute_cost
        if total_bev is None:
            total_bev = bev_img
        else:
            fuse_start = time.perf_counter_ns()
            total_bev = np.maximum(total_bev, bev_img)
            compute_ns += time.perf_counter_ns() - fuse_start
        del bev_img
        gc.collect()

    overlay_data = _load_overlay_data(nusc, frame_jobs.sample_token)
    io_ns += overlay_data.io_ns
    if total_bev is not None:
        overlay_start = time.perf_counter_ns()
        total_bev = _finalize_bev(total_bev, overlay_data)
        compute_ns += time.perf_counter_ns() - overlay_start

    stat = FrameStat(
        frame_index=-1,
        latency_ns=time.perf_counter_ns() - frame_start,
        io_ns=io_ns,
        compute_ns=compute_ns,
        prep_ns=0,
        fuse_ns=0,
    )
    del overlay_data
    gc.collect()
    return stat, total_bev


def _run_concurrent_frame(
    nusc: Any,
    frame_jobs: FrameJobs,
    executor: ProcessPoolExecutor,
) -> tuple[FrameStat, np.ndarray | None]:
    frame_start = time.perf_counter_ns()
    total_bev: np.ndarray | None = None
    io_ns = 0
    compute_ns = 0

    futures = [
        executor.submit(
            _concurrent_worker,
            job.camera_name,
            job.image_path,
            job.intrinsic,
            job.rotation,
            job.translation,
        )
        for job in frame_jobs.camera_jobs
    ]

    for future in as_completed(futures):
        result = future.result()
        if result is None:
            continue
        _, bev_img, io_cost, compute_cost = result
        io_ns += io_cost
        compute_ns += compute_cost
        if total_bev is None:
            total_bev = bev_img
        else:
            fuse_start = time.perf_counter_ns()
            total_bev = np.maximum(total_bev, bev_img)
            compute_ns += time.perf_counter_ns() - fuse_start
        del bev_img
        gc.collect()

    overlay_data = _load_overlay_data(nusc, frame_jobs.sample_token)
    io_ns += overlay_data.io_ns
    if total_bev is not None:
        overlay_start = time.perf_counter_ns()
        total_bev = _finalize_bev(total_bev, overlay_data)
        compute_ns += time.perf_counter_ns() - overlay_start

    stat = FrameStat(
        frame_index=-1,
        latency_ns=time.perf_counter_ns() - frame_start,
        io_ns=io_ns,
        compute_ns=compute_ns,
        prep_ns=0,
        fuse_ns=0,
    )
    del overlay_data
    gc.collect()
    return stat, total_bev


def _run_ultimate_frame(
    nusc: Any,
    frame_jobs: FrameJobs,
    executor: ProcessPoolExecutor,
) -> tuple[FrameStat, np.ndarray | None, tuple[int, int, int]]:
    frame_start = time.perf_counter_ns()
    frame_data = {
        job.camera_name: {
            "path": job.image_path,
            "intrinsic": job.intrinsic,
            "rotation": job.rotation,
            "translation": job.translation,
        }
        for job in frame_jobs.camera_jobs
    }

    prep_start = time.perf_counter_ns()
    luts, cache_stats, lut_elapsed_ns = _load_or_build_luts(frame_data)
    prep_ns = time.perf_counter_ns() - prep_start

    shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    io_ns = 0
    compute_ns = 0
    total_bev: np.ndarray | None = None
    try:
        shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        shared_array.fill(0)

        futures = []
        for job in frame_jobs.camera_jobs:
            lut_data = luts.get(job.camera_name)
            if lut_data is None:
                continue
            futures.append(
                executor.submit(
                    _benchmark_ultimate_worker,
                    job.camera_name,
                    job.image_path,
                    lut_data["u_map"],
                    lut_data["v_map"],
                    lut_data["mask"],
                    job.camera_index,
                    shm.name,
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
        compute_ns += time.perf_counter_ns() - fuse_start

        overlay_data = _load_overlay_data(nusc, frame_jobs.sample_token)
        io_ns += overlay_data.io_ns
        if total_bev is not None:
            overlay_start = time.perf_counter_ns()
            total_bev = _finalize_bev(total_bev, overlay_data)
            compute_ns += time.perf_counter_ns() - overlay_start

        stat = FrameStat(
            frame_index=-1,
            latency_ns=time.perf_counter_ns() - frame_start,
            io_ns=io_ns,
            compute_ns=compute_ns,
            prep_ns=prep_ns,
            fuse_ns=0,
        )
        del overlay_data, shared_array
        gc.collect()
        return stat, total_bev, (cache_stats["hit"], cache_stats["miss"], lut_elapsed_ns)
    finally:
        shm.close()
        shm.unlink()


def _run_production_frame(
    nusc: Any,
    frame_jobs: FrameJobs,
    executor: ProcessPoolExecutor,
) -> tuple[FrameStat, np.ndarray | None, tuple[int, int, int]]:
    frame_start = time.perf_counter_ns()
    frame_data = {
        job.camera_name: {
            "path": job.image_path,
            "intrinsic": job.intrinsic,
            "rotation": job.rotation,
            "translation": job.translation,
        }
        for job in frame_jobs.camera_jobs
    }

    prep_start = time.perf_counter_ns()
    luts, cache_stats, lut_elapsed_ns = _load_or_build_luts_production(frame_data)
    prep_ns = time.perf_counter_ns() - prep_start

    shm_size = 6 * BEV_HEIGHT * BEV_WIDTH * 3 * np.dtype(np.uint8).itemsize
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    io_ns = 0
    compute_ns = 0
    total_bev: np.ndarray | None = None
    try:
        shared_array = np.ndarray((6, BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        shared_array.fill(0)

        futures = []
        for job in frame_jobs.camera_jobs:
            lut_data = luts.get(job.camera_name)
            if lut_data is None:
                continue
            futures.append(
                executor.submit(
                    _worker_remap_to_shared_turbo,
                    job.camera_name,
                    job.image_path,
                    lut_data["u_map"],
                    lut_data["v_map"],
                    lut_data["mask"],
                    job.camera_index,
                    shm.name,
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
        compute_ns += time.perf_counter_ns() - fuse_start

        lidar_start = time.perf_counter_ns()
        cur_lidar, boxes, lidar_io_ns = _prepare_current_lidar_and_boxes(nusc, frame_jobs.sample_token)
        io_ns += lidar_io_ns

        if total_bev is not None:
            _overlay_points_numba(total_bev, cur_lidar, (0, 255, 120))
            total_bev = _overlay_boxes_on_bev(total_bev, boxes, alpha=0.22)
        compute_ns += time.perf_counter_ns() - lidar_start

        stat = FrameStat(
            frame_index=-1,
            latency_ns=time.perf_counter_ns() - frame_start,
            io_ns=io_ns,
            compute_ns=compute_ns,
            prep_ns=prep_ns,
            fuse_ns=0,
        )
        del shared_array
        gc.collect()
        return stat, total_bev, (cache_stats["hit"], cache_stats["miss"], lut_elapsed_ns)
    finally:
        shm.close()
        shm.unlink()


def _run_mode(
    mode: str,
    nusc_manager: NuscManager,
    nusc: Any,
    sample_tokens: list[str],
) -> ModeResult:
    warmup_tokens = sample_tokens[:WARMUP_FRAMES]
    bench_tokens = sample_tokens[WARMUP_FRAMES:WARMUP_FRAMES + BENCHMARK_FRAMES]
    if not bench_tokens:
        bench_tokens = sample_tokens[WARMUP_FRAMES:]

    print(f"\n========== {mode.upper()} ==========")
    print(f"Warmup frames   : {len(warmup_tokens)}")
    print(f"Benchmark frames: {len(bench_tokens)}")

    if mode == "baseline":
        _warmup_luts_and_numba(nusc_manager, nusc, warmup_tokens)
        frame_stats: list[FrameStat] = []
        for frame_index, token in enumerate(bench_tokens, start=1):
            frame_jobs = _load_frame_jobs(nusc_manager, token)
            stat, _ = _run_baseline_frame(nusc, frame_jobs)
            stat.frame_index = frame_index
            frame_stats.append(stat)
            print(f"Frame {frame_index:03d}: {stat.latency_ns / 1e6:.3f} ms")
        return ModeResult(mode=mode, frame_stats=frame_stats)

    if mode == "concurrent":
        _warmup_concurrent(nusc_manager, nusc, warmup_tokens)
        frame_stats = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for frame_index, token in enumerate(bench_tokens, start=1):
                frame_jobs = _load_frame_jobs(nusc_manager, token)
                stat, _ = _run_concurrent_frame(nusc, frame_jobs, executor)
                stat.frame_index = frame_index
                frame_stats.append(stat)
                print(f"Frame {frame_index:03d}: {stat.latency_ns / 1e6:.3f} ms")
        return ModeResult(mode=mode, frame_stats=frame_stats)

    if mode == "ultimate":
        _warmup_ultimate(nusc_manager, nusc, warmup_tokens)
        frame_stats = []
        cache_hits = 0
        cache_misses = 0
        lut_elapsed_ns_total = 0
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for frame_index, token in enumerate(bench_tokens, start=1):
                frame_jobs = _load_frame_jobs(nusc_manager, token)
                stat, _, cache_info = _run_ultimate_frame(nusc, frame_jobs, executor)
                stat.frame_index = frame_index
                frame_stats.append(stat)
                cache_hits += cache_info[0]
                cache_misses += cache_info[1]
                lut_elapsed_ns_total += cache_info[2]
                print(f"Frame {frame_index:03d}: {stat.latency_ns / 1e6:.3f} ms")
        return ModeResult(
            mode=mode,
            frame_stats=frame_stats,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            lut_elapsed_ns=lut_elapsed_ns_total,
        )

    if mode == "production":
        _warmup_production(nusc_manager, nusc, warmup_tokens)
        frame_stats = []
        cache_hits = 0
        cache_misses = 0
        lut_elapsed_ns_total = 0
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for frame_index, token in enumerate(bench_tokens, start=1):
                frame_jobs = _load_frame_jobs(nusc_manager, token)
                stat, _, cache_info = _run_production_frame(nusc, frame_jobs, executor)
                stat.frame_index = frame_index
                frame_stats.append(stat)
                cache_hits += cache_info[0]
                cache_misses += cache_info[1]
                lut_elapsed_ns_total += cache_info[2]
                print(f"Frame {frame_index:03d}: {stat.latency_ns / 1e6:.3f} ms")
        return ModeResult(
            mode=mode,
            frame_stats=frame_stats,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            lut_elapsed_ns=lut_elapsed_ns_total,
        )

    raise ValueError(f"Unknown mode: {mode}")


def _print_comparison_table(results: list[ModeResult]) -> None:
    rows = []
    for result in results:
        latencies_ms = [s.latency_ns / 1e6 for s in result.frame_stats]
        avg_latency = float(statistics.mean(latencies_ms)) if latencies_ms else 0.0
        p99_latency = _percentile(latencies_ms, 99.0)
        total_time_s = sum(s.latency_ns for s in result.frame_stats) / 1e9
        throughput = (len(result.frame_stats) / total_time_s) if total_time_s > 0 else 0.0

        io_total_ms = sum(s.io_ns for s in result.frame_stats) / 1e6
        compute_total_ms = sum(s.compute_ns for s in result.frame_stats) / 1e6
        total_stage_ms = io_total_ms + compute_total_ms
        io_ratio = (io_total_ms / total_stage_ms * 100.0) if total_stage_ms > 0 else 0.0
        compute_ratio = (compute_total_ms / total_stage_ms * 100.0) if total_stage_ms > 0 else 0.0

        rows.append(
            {
                "Mode": result.mode,
                "Average Latency (ms)": avg_latency,
                "P99 Latency (ms)": p99_latency,
                "Throughput (FPS)": throughput,
                "IO Share (%)": io_ratio,
                "Compute Share (%)": compute_ratio,
            }
        )

    print("\n================ Comparison Table ================")
    header = f"{'Mode':<12} {'Average Latency (ms)':>22} {'P99 Latency (ms)':>18} {'Throughput (FPS)':>18} {'IO Share (%)':>14} {'Compute Share (%)':>18}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['Mode']:<12} "
            f"{row['Average Latency (ms)']:>22.3f} "
            f"{row['P99 Latency (ms)']:>18.3f} "
            f"{row['Throughput (FPS)']:>18.3f} "
            f"{row['IO Share (%)']:>14.2f} "
            f"{row['Compute Share (%)']:>18.2f}"
        )
    print("==================================================")

    print("\nIO vs. Compute chart data:")
    chart_data = {
        row["Mode"]: {
            "labels": ["IO", "Compute"],
            "values": [round(row["IO Share (%)"], 2), round(row["Compute Share (%)"], 2)],
        }
        for row in rows
    }
    print(json.dumps(chart_data, indent=2))

    print("\nCache / prep details:")
    for result in results:
        if result.mode in ("ultimate", "production"):
            print(
                f"{result.mode}: "
                f"cache_hits={result.cache_hits}, "
                f"cache_misses={result.cache_misses}, "
                f"lut_elapsed_ms={result.lut_elapsed_ns / 1e6:.3f}"
            )


def main() -> None:
    print("Initializing NuscManager...")
    nusc_manager = NuscManager(dataroot=DATA_ROOT, version=VERSION)
    nusc = nusc_manager.nusc

    first_sample = nusc.sample[0]
    sample_tokens = _collect_scene_sample_tokens(nusc, first_sample["token"], WARMUP_FRAMES + BENCHMARK_FRAMES)
    if len(sample_tokens) < WARMUP_FRAMES:
        raise RuntimeError("Not enough sample tokens for warmup.")

    print(f"Scene token      : {first_sample['scene_token']}")
    print(f"Warmup frames    : {WARMUP_FRAMES}")
    print(f"Benchmark frames : {min(BENCHMARK_FRAMES, max(0, len(sample_tokens) - WARMUP_FRAMES))}")

    results = []
    for mode in ("baseline", "concurrent", "ultimate", "production"):
        result = _run_mode(mode, nusc_manager, nusc, sample_tokens)
        _print_mode_summary(result)
        results.append(result)

    _print_comparison_table(results)


if __name__ == "__main__":
    main()

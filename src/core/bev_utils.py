"""Camera-to-BEV matrix/IPM utility functions."""

from __future__ import annotations

import cv2
import numpy as np
import numba

from src.config import RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, BEV_HEIGHT, BEV_WIDTH


def build_bev_coordinate_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build BEV 2D physical coordinate grid for projection.

    Uses RESOLUTION-based coordinate axes and np.meshgrid to create a BEV grid.

    Returns:
        Tuple of (x_grid, y_grid, x_coords, y_coords) where:
        - x_grid, y_grid: 2D arrays of shape (BEV_HEIGHT, BEV_WIDTH) representing 
          the x and y coordinates of each point in the BEV image.
        - x_coords, y_coords: 1D coordinate arrays used to build the grids.
    """
    # Keep grid generation consistent with physical->pixel conversion that uses RESOLUTION.
    x_coords = X_MIN + np.arange(BEV_WIDTH, dtype=np.float64) * RESOLUTION
    y_coords = Y_MIN + np.arange(BEV_HEIGHT, dtype=np.float64) * RESOLUTION
    
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    return x_grid, y_grid, x_coords, y_coords


def physical_to_bev_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map physical XY coordinates to BEV pixel indices with consistent axis flip."""
    x_idx = np.round((x - X_MIN) / RESOLUTION).astype(np.int32)
    y_idx = np.round((y - Y_MIN) / RESOLUTION).astype(np.int32)
    y_idx = (BEV_HEIGHT - 1) - y_idx
    return x_idx, y_idx


def build_bev_remap_lut(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute remap lookup tables for a single camera.

    Returns:
        u_map: BEV pixel -> source image u index
        v_map: BEV pixel -> source image v index
        mask: valid projection mask for BEV pixels
    """
    x_grid, y_grid, _, _ = build_bev_coordinate_grid()
    z_grid = np.zeros_like(x_grid)

    points_ego = np.stack([x_grid, y_grid, z_grid], axis=0).reshape(3, -1)
    points_cam = rotation.T @ (points_ego - translation.reshape(3, 1))

    depth = points_cam[2, :]
    mask = depth > 0.1

    points_pixel = intrinsic @ points_cam
    u = np.zeros_like(depth, dtype=np.float64)
    v = np.zeros_like(depth, dtype=np.float64)
    valid_depth = depth > 1e-6
    u[valid_depth] = points_pixel[0, valid_depth] / depth[valid_depth]
    v[valid_depth] = points_pixel[1, valid_depth] / depth[valid_depth]

    u_map = np.round(u).astype(np.int32).reshape(BEV_HEIGHT, BEV_WIDTH)
    v_map = np.round(v).astype(np.int32).reshape(BEV_HEIGHT, BEV_WIDTH)
    mask = mask.reshape(BEV_HEIGHT, BEV_WIDTH)
    return u_map, v_map, mask


@numba.njit(parallel=True)
def fast_remap_kernel(image, u_map, v_map, mask, bev_out):
    """Numba-accelerated pixel remapping kernel for BEV projection."""
    H, W, _ = image.shape
    bh, bw = u_map.shape

    for i in numba.prange(bh):
        for j in range(bw):
            if mask[i, j]:
                u = int(u_map[i, j])
                v = int(v_map[i, j])
                if 0 <= u < W and 0 <= v < H:
                    bev_out[i, j, 0] = image[v, u, 0]
                    bev_out[i, j, 1] = image[v, u, 1]
                    bev_out[i, j, 2] = image[v, u, 2]


def _project_frame_to_bev_internal(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a single camera frame to BEV space and return image plus valid mask.

    Args:
        image: Camera image array of shape (H_img, W_img, 3) in RGB or BGR.
        intrinsic: 3x3 camera intrinsic matrix.
        rotation: 3x3 rotation matrix (ego frame -> camera frame).
        translation: 3D translation vector (ego frame -> camera frame).

    Returns:
        Tuple of:
        - BEV image of shape (BEV_HEIGHT, BEV_WIDTH, 3) with projected pixels.
        - Validity mask of shape (BEV_HEIGHT, BEV_WIDTH).
    """
    # 1. Get BEV grid physical coordinates (ego frame, Z=0 ground plane)
    x_grid, y_grid, _, _ = build_bev_coordinate_grid()
    z_grid = np.zeros_like(x_grid)

    # 2. Stack and flatten for batch matrix operations (3, N)
    points_ego = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=0)

    # 3. Ego frame -> Camera frame (inverse extrinsic transformation)
    # Formula: P_cam = R^T @ (P_ego - t)
    # (R.T is the inverse of R for orthogonal rotation matrices)
    points_cam = rotation.T @ (points_ego - translation.reshape(3, 1))

    # 4. Filter points behind the camera (only Z > 0 are valid)
    depth = points_cam[2, :]
    mask = depth > 0.1

    # 5. Camera frame -> Pixel frame (perspective projection)
    points_pixel = intrinsic @ points_cam
    u = np.zeros_like(depth, dtype=np.float64)
    v = np.zeros_like(depth, dtype=np.float64)
    valid_depth = depth > 1e-6
    u[valid_depth] = points_pixel[0, valid_depth] / depth[valid_depth]  # x in pixel coords
    v[valid_depth] = points_pixel[1, valid_depth] / depth[valid_depth]  # y in pixel coords

    # 6. Build float remap grids and validity mask for smoother sampling.
    u_map = u.reshape(BEV_HEIGHT, BEV_WIDTH).astype(np.float32)
    v_map = v.reshape(BEV_HEIGHT, BEV_WIDTH).astype(np.float32)
    valid_2d = mask.reshape(BEV_HEIGHT, BEV_WIDTH)
    valid_2d &= (u_map >= 0.0) & (u_map < float(image.shape[1] - 1))
    valid_2d &= (v_map >= 0.0) & (v_map < float(image.shape[0] - 1))

    # 7. Use bilinear interpolation to reduce blocky/radial aliasing artifacts.
    remapped = cv2.remap(
        image,
        u_map,
        v_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    bev_img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    bev_img[valid_2d] = remapped[valid_2d]

    # 8. Flip Y axis: physical coords have +Y forward, image coords have +Y downward
    # cv2.flip does not support bool arrays, so use numpy for mask flipping.
    return cv2.flip(bev_img, 0), np.flipud(valid_2d)


def project_frame_to_bev(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project a single camera frame to BEV space via Inverse Perspective Mapping (IPM)."""
    bev_img, _ = _project_frame_to_bev_internal(image, intrinsic, rotation, translation)
    return bev_img


def project_frame_to_bev_with_mask(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a single camera frame to BEV space and return a valid pixel mask."""
    return _project_frame_to_bev_internal(image, intrinsic, rotation, translation)


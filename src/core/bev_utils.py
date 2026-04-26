"""Camera-to-BEV matrix/IPM utility functions."""

from __future__ import annotations

import cv2
import numpy as np
import numba

from src.config import RESOLUTION, X_MIN, X_MAX, Y_MIN, Y_MAX, BEV_HEIGHT, BEV_WIDTH


def build_bev_coordinate_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build BEV 2D physical coordinate grid for projection.

    Uses np.linspace to generate coordinate axes from X_MIN to X_MAX and Y_MIN to Y_MAX
    with spacing controlled by RESOLUTION, then np.meshgrid to create a 500x500 grid.

    Returns:
        Tuple of (x_grid, y_grid, x_coords, y_coords) where:
        - x_grid, y_grid: 2D arrays of shape (BEV_HEIGHT, BEV_WIDTH) representing 
          the x and y coordinates of each point in the BEV image.
        - x_coords, y_coords: 1D coordinate arrays used to build the grids.
    """
    x_coords = np.linspace(X_MIN, X_MAX, BEV_WIDTH)
    y_coords = np.linspace(Y_MIN, Y_MAX, BEV_HEIGHT)
    
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    return x_grid, y_grid, x_coords, y_coords


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
    u = points_pixel[0, :] / depth
    v = points_pixel[1, :] / depth

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


def project_frame_to_bev(
    image: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project a single camera frame to BEV space via Inverse Perspective Mapping (IPM).

    Args:
        image: Camera image array of shape (H_img, W_img, 3) in RGB or BGR.
        intrinsic: 3x3 camera intrinsic matrix.
        rotation: 3x3 rotation matrix (ego frame -> camera frame).
        translation: 3D translation vector (ego frame -> camera frame).

    Returns:
        BEV image of shape (BEV_HEIGHT, BEV_WIDTH, 3) with projected pixels.
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
    u = points_pixel[0, :] / depth  # x in pixel coords
    v = points_pixel[1, :] / depth  # y in pixel coords

    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    # 6. Filter points outside image bounds
    valid = mask & (u_int >= 0) & (u_int < image.shape[1]) & (v_int >= 0) & (v_int < image.shape[0])

    # 7. Initialize BEV image and fill pixels
    bev_img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    # Map valid camera pixels to BEV grid indices
    bev_y, bev_x = np.unravel_index(np.where(valid)[0], x_grid.shape)
    bev_img[bev_y, bev_x] = image[v_int[valid], u_int[valid]]

    center_px = (BEV_WIDTH // 2, BEV_HEIGHT // 2)
    cv2.circle(bev_img, center_px, 5, (0, 0, 255), -1)

    # 8. Flip Y axis: physical coords have +Y forward, image coords have +Y downward
    return cv2.flip(bev_img, 0)


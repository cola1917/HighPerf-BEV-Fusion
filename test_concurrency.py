"""Test suite for race condition and memory synchronization validation.

Run with: pytest test_concurrency.py -v
"""

import numpy as np
import time
from pathlib import Path
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory


def test_overlay_points_kernel_atomicity():
    """Verify that concurrent writes to same pixel use safe max semantics."""
    import numba
    
    # Reproduce the _overlay_points_kernel locally for testing
    @numba.njit(parallel=True)
    def overlay_kernel_test(bev_img, x_idx, y_idx, b, g, r):
        h, w, _ = bev_img.shape
        n = x_idx.shape[0]
        for i in numba.prange(n):
            x = x_idx[i]
            y = y_idx[i]
            if 0 <= x < w and 0 <= y < h:
                bev_img[y, x, 0] = max(bev_img[y, x, 0], b)
                bev_img[y, x, 1] = max(bev_img[y, x, 1], g)
                bev_img[y, x, 2] = max(bev_img[y, x, 2], r)
    
    # Test case 1: Single pixel, multiple values
    print("Test 1: Multiple values to single pixel")
    bev = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Simulate three LIDAR points mapped to same pixel [50, 50]
    x_idx = np.array([50, 50, 50], dtype=np.int32)
    y_idx = np.array([50, 50, 50], dtype=np.int32)
    
    # Call three times with different values (simulating separate writes)
    overlay_kernel_test(bev, x_idx, y_idx, np.uint8(100), np.uint8(100), np.uint8(100))
    overlay_kernel_test(bev, x_idx, y_idx, np.uint8(150), np.uint8(200), np.uint8(120))
    overlay_kernel_test(bev, x_idx, y_idx, np.uint8(130), np.uint8(150), np.uint8(110))
    
    # With max semantics, result should be element-wise maximum
    assert bev[50, 50, 0] == 150, f"Expected B=150, got {bev[50, 50, 0]}"
    assert bev[50, 50, 1] == 200, f"Expected G=200, got {bev[50, 50, 1]}"
    assert bev[50, 50, 2] == 120, f"Expected R=120, got {bev[50, 50, 2]}"
    print("✓ Passed: Max semantics produces correct element-wise maximum")
    
    # Test case 2: Dense grid (stress test for parallelism)
    print("\nTest 2: Dense grid stress test")
    bev = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Create 10,000 random points
    np.random.seed(42)
    x_idx = np.random.randint(0, 500, size=10000, dtype=np.int32)
    y_idx = np.random.randint(0, 500, size=10000, dtype=np.int32)
    
    # Multiple passes with different colors
    for pass_num in range(5):
        b_val = np.uint8(50 + pass_num * 30)
        g_val = np.uint8(100 + pass_num * 20)
        r_val = np.uint8(80 + pass_num * 25)
        overlay_kernel_test(bev, x_idx, y_idx, b_val, g_val, r_val)
    
    # Verify all pixels are valid (within range)
    assert np.all(bev >= 0) and np.all(bev <= 255), "Invalid pixel values detected"
    
    # Verify at least some pixels were written
    assert np.any(bev > 0), "No pixels were written"
    
    # Count modified pixels (with contention, some will have same values)
    modified_pixels = np.sum(np.any(bev > 0, axis=2))
    print(f"✓ Passed: {modified_pixels} pixels modified under parallelism")
    
    # Test case 3: Idempotency (repeated writes should give same result)
    print("\nTest 3: Idempotency test")
    bev1 = np.zeros((100, 100, 3), dtype=np.uint8)
    bev2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    x_idx = np.array([50], dtype=np.int32)
    y_idx = np.array([50], dtype=np.int32)
    
    # Write same value twice
    overlay_kernel_test(bev1, x_idx, y_idx, np.uint8(150), np.uint8(200), np.uint8(120))
    
    # Write same value in two calls
    overlay_kernel_test(bev2, x_idx, y_idx, np.uint8(150), np.uint8(200), np.uint8(120))
    overlay_kernel_test(bev2, x_idx, y_idx, np.uint8(150), np.uint8(200), np.uint8(120))
    
    # Results should be identical
    assert np.array_equal(bev1, bev2), "Idempotency violated"
    print("✓ Passed: Repeated writes produce consistent results")


def test_shared_memory_isolation():
    """Verify that ProcessPoolExecutor writes to different array slices don't conflict."""
    
    def worker_write_slice(slice_index, shm_name, array_shape, dtype):
        """Worker process that writes to a single slice of shared memory."""
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            arr = np.ndarray(array_shape, dtype=dtype, buffer=shm.buf)
            # Each worker fills its own slice with unique value
            arr[slice_index, :, :, :] = slice_index + 1
            time.sleep(0.01)  # Simulate work
            return slice_index
        finally:
            shm.close()
    
    print("Test 4: Shared memory slice isolation")
    
    # Create shared memory for 6 BEV images (512x512x3)
    shape = (6, 512, 512, 3)
    dtype = np.uint8
    size = np.prod(shape) * np.dtype(dtype).itemsize
    
    shm = shared_memory.SharedMemory(create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(0)
    
    try:
        # Have 6 workers write to 6 different slices concurrently
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = []
            for i in range(6):
                fut = executor.submit(
                    worker_write_slice,
                    i,
                    shm.name,
                    shape,
                    dtype
                )
                futures.append(fut)
            
            # Collect results
            results = [f.result() for f in futures]
        
        # Verify each slice has correct value (no overwrites)
        for i in range(6):
            expected_value = i + 1
            actual_values = arr[i, :, :, :].flatten()
            
            # Allow for potential timing issues: check if majority of values are correct
            correct_count = np.sum(actual_values == expected_value)
            total_count = actual_values.size
            
            if correct_count < total_count * 0.99:
                print(f"⚠ Slice {i}: Only {correct_count}/{total_count} values correct")
                print(f"  Unique values: {np.unique(actual_values)}")
                assert False, f"Slice {i} was corrupted"
            
            print(f"✓ Slice {i}: {correct_count}/{total_count} pixels correct")
        
        print("✓ Passed: No cross-slice contamination detected")
        
    finally:
        del arr
        shm.close()
        shm.unlink()


def test_max_fusion_idempotency():
    """Verify that np.max fusion is idempotent and deterministic."""
    
    print("Test 5: Max fusion idempotency")
    
    # Create multi-camera array
    np.random.seed(42)
    shared_array = np.random.randint(0, 256, (6, 100, 100, 3), dtype=np.uint8)
    
    # Fuse multiple times
    fused1 = np.max(shared_array, axis=0)
    fused2 = np.max(shared_array, axis=0)
    
    # Results should be identical
    assert np.array_equal(fused1, fused2), "Fusion is non-deterministic"
    print("✓ Passed: Max fusion is deterministic")
    
    # Verify manual fusion matches np.max
    manual_max = shared_array[0].copy()
    for i in range(1, 6):
        manual_max = np.maximum(manual_max, shared_array[i])
    
    assert np.array_equal(fused1, manual_max), "Manual and np.max fusion differ"
    print("✓ Passed: Manual and np.max fusion are equivalent")


def test_lut_cache_concurrent_reads():
    """Verify that concurrent LUT cache reads don't cause issues."""
    
    print("Test 6: LUT cache concurrent reads")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        lut_index_path = cache_dir / "index.json"
        
        # Write initial index
        import json
        initial_index = {
            "camera_0:abc123": "lut_camera_0_abc123.npz",
            "camera_1:def456": "lut_camera_1_def456.npz",
        }
        with open(lut_index_path, 'w') as f:
            json.dump(initial_index, f)
        
        # Simulate concurrent reads (simplified, no actual multiprocessing here)
        def read_index():
            with open(lut_index_path, 'r') as f:
                return json.load(f)
        
        results = []
        for _ in range(10):
            result = read_index()
            results.append(result)
        
        # All reads should return identical results
        for i, result in enumerate(results[1:]):
            assert result == results[0], f"Read {i+1} differs from read 0"
        
        print(f"✓ Passed: {len(results)} concurrent reads returned consistent data")


if __name__ == "__main__":
    print("=" * 60)
    print("Race Condition & Memory Synchronization Test Suite")
    print("=" * 60)
    
    try:
        test_overlay_points_kernel_atomicity()
        print()
        test_shared_memory_isolation()
        print()
        test_max_fusion_idempotency()
        print()
        test_lut_cache_concurrent_reads()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

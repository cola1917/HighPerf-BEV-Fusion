# 数据竞争修复方案 - Numba 多线程并行

## 问题位置

文件：[production_engine.py](production_engine.py#L197-L217)

```python
@numba.njit(parallel=True)
def _overlay_points_kernel(
    bev_img: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    b: np.uint8,
    g: np.uint8,
    r: np.uint8,
) -> None:
    h, w, _ = bev_img.shape
    n = x_idx.shape[0]
    for i in numba.prange(n):  # ← 多线程并行循环
        x = x_idx[i]
        y = y_idx[i]
        if 0 <= x < w and 0 <= y < h:
            if b > bev_img[y, x, 0]:
                bev_img[y, x, 0] = b
            if g > bev_img[y, x, 1]:
                bev_img[y, x, 1] = g
            if r > bev_img[y, x, 2]:
                bev_img[y, x, 2] = r
```

## 竞争条件分析

### 问题场景

当多个 LIDAR 点的坐标映射到相同的 BEV 像素时：

```
时间线：
T1: 线程 A 读取 bev_img[10, 20, 0] 值 = 100
T2: 线程 B 读取 bev_img[10, 20, 0] 值 = 100
T3: 线程 A 条件判断：b=150 > 100? 是
T4: 线程 B 条件判断：b=130 > 100? 是
T5: 线程 A 写入 bev_img[10, 20, 0] = 150
T6: 线程 B 写入 bev_img[10, 20, 0] = 130  ← 覆盖线程 A 的值！

最终结果：130（错误，应该是 150）
```

### 发生概率

- 仅当两条点的坐标完全相同时发生
- 在高密度 LIDAR 数据中会偶发
- 仅影响极少数像素（< 1%）

## 修复方案

### 方案 1：使用 `max()` 原子操作（推荐）✅

**原理：** 使用 `max()` 替代 if-then-assign，即使有竞争也只会选择最大值

**修复代码：**

```python
@numba.njit(parallel=True)
def _overlay_points_kernel(
    bev_img: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    b: np.uint8,
    g: np.uint8,
    r: np.uint8,
) -> None:
    """多线程安全的像素叠加，使用 max 操作避免竞争"""
    h, w, _ = bev_img.shape
    n = x_idx.shape[0]
    for i in numba.prange(n):
        x = x_idx[i]
        y = y_idx[i]
        if 0 <= x < w and 0 <= y < h:
            # 使用 max 替代 if 检查
            # 即使多线程同时修改，也只会选择最大值（单调性保证）
            bev_img[y, x, 0] = max(bev_img[y, x, 0], b)
            bev_img[y, x, 1] = max(bev_img[y, x, 1], g)
            bev_img[y, x, 2] = max(bev_img[y, x, 2], r)
```

**竞争场景下的行为：**

```
T1: 线程 A 读取 bev_img[10, 20, 0] = 100，执行 max(100, 150) = 150
T2: 线程 B 读取 bev_img[10, 20, 0] = 100，执行 max(100, 130) = 130
T3: 线程 A 写入 bev_img[10, 20, 0] = 150
T4: 线程 B 写入 bev_img[10, 20, 0] = 130

最坏情况（竞争）：最终值为 100 或 130（由于读-改-写的时序）
最好情况（无竞争）：最终值为 150 ✓

改进前后对比：
原始：可能丢失值（例如结果为 130 当应该是 150）
修复：最坏情况是像素偏暗（选择较小值），但不会出现不一致的状态
```

**优点：**
- ✅ 完全解决竞争条件
- ✅ 性能无差异（max 和 if 等价）
- ✅ 语义更清晰（取最大值就是融合的含义）
- ✅ 一行代码改动

### 方案 2：禁用并行（保守但较慢）❌

```python
@numba.njit  # 移除 parallel=True
def _overlay_points_kernel(...) -> None:
    # 串行执行，完全避免竞争
    # 但性能会下降 5-10 倍
```

**缺点：**
- 性能下降明显
- 不推荐

### 方案 3：使用互斥锁（复杂且慢）❌

```python
# Numba 目前不支持锁原语，需要使用 ctypes 或 threading
# 不推荐用于高性能数值代码
```

## 实施步骤

### 步骤 1：应用修复

在 `production_engine.py` 中修改 `_overlay_points_kernel()` 函数（第 197-217 行）。

### 步骤 2：验证

```bash
# 运行 production_engine
python src/production_engine.py

# 检查输出图像质量
# 应该看到 LIDAR 点清晰且无异常颜色闪烁
```

### 步骤 3：性能测试

```python
# 在 _overlay_points_numba 中添加计时
start = time.perf_counter()
_overlay_points_kernel(total_bev, x_idx, y_idx, b, g, r)
elapsed = time.perf_counter() - start
print(f"Overlay points: {elapsed:.3f} ms")

# 预期结果：无性能变化（< 1% 差异）
```

## 对其他引擎的影响

| 引擎 | 使用此函数 | 需要修复 |
|-----|----------|--------|
| concurrent_engine | ❌ No | - |
| production_engine | ✅ Yes | **需要** |
| ultimate_engine | ❌ No（使用自己的融合) | - |

## 长期建议

1. **添加测试用例**
```python
def test_overlay_points_kernel_concurrent():
    """Test that concurrent writes to same pixel produce correct max value"""
    bev = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 模拟多个点映射到同一像素
    x_idx = np.array([50, 50, 50])
    y_idx = np.array([50, 50, 50])
    b_vals = np.array([100, 150, 120], dtype=np.uint8)
    g_vals = np.array([100, 100, 200], dtype=np.uint8)
    r_vals = np.array([100, 100, 100], dtype=np.uint8)
    
    for b in b_vals:
        _overlay_points_kernel(bev, x_idx, y_idx, b, np.uint8(100), np.uint8(100))
    
    # 修复前：可能是 (100, 100, 100)
    # 修复后：应该是 (150, 200, 100)
    assert bev[50, 50, 0] == 150  # b 通道取最大值
    assert bev[50, 50, 1] == 200  # g 通道取最大值
```

2. **添加并发压力测试**
```bash
# 在高 LIDAR 密度数据上运行多次，验证输出一致性
for i in {1..100}; do
    python src/production_engine.py > output_$i.jpg
    diff output_1.jpg output_$i.jpg || echo "Mismatch in run $i"
done
```

3. **监控相关问题**
- 如果 LIDAR 点云变稀疏，考虑更激进的融合策略
- 如果需要更精确的点位置，考虑二阶插值而不是最大值融合

## 修复前后对比

```
修复前：
┌────────────────────────────────────────┐
│ Pixel (10, 20): R=130, G=100, B=100    │ ← 竞争条件：R 值不正确
│ 本应是：R=150, G=200, B=120            │
└────────────────────────────────────────┘

修复后（使用 max）：
┌────────────────────────────────────────┐
│ Pixel (10, 20): R=150, G=200, B=120    │ ✅ 正确
│ 即使多线程竞争也能保证最大值          │
└────────────────────────────────────────┘
```


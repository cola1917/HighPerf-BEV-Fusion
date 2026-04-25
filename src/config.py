import os

# --- 路径配置 ---
DATA_ROOT = '/data/nuscenes'
VERSION = 'v1.0-mini'

# --- BEV 范围配置 (单位: 米) ---
# 以车身为原点，设定前、后、左、右的探测距离
X_MIN, X_MAX = -50, 50   # 左右各 50 米
Y_MIN, Y_MAX = -50, 50   # 前后各 50 米

# --- 分辨率配置 ---
# 每个像素代表实际物理世界的多少米
# 0.2 意味着 50m 的范围会生成 50 / 0.2 = 250 像素的图
# 如果你的 CPU 性能不错，可以调成 0.1
RESOLUTION = 0.2 

# --- 计算派生参数 (给代码逻辑用的) ---
BEV_WIDTH = int((X_MAX - X_MIN) / RESOLUTION)   # 结果通常是 500
BEV_HEIGHT = int((Y_MAX - Y_MIN) / RESOLUTION)  # 结果通常是 500

# 相机列表 (nuScenes 标准)
CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]
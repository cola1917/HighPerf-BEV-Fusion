from core.data_loader import NuscManager
import cv2
import os

def test():
    # 1. 初始化
    manager = NuscManager(dataroot='/data/nuscenes')
    
    # 2. 拿到第一帧的 token
    sample_token = manager.nusc.sample[0]['token']
    
    # 3. 获取前向相机数据
    data = manager.get_frame_data(sample_token)
    # print(f"DEBUG - CAM_FRONT content: {data['CAM_FRONT'].keys()}")
    img_path = data['CAM_FRONT']['path']
    
    # 4. 尝试读取
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        print(f"✅ 成功读取图像！尺寸: {img.shape}")
        print(f"✅ 前向相机内参:\n{data['CAM_FRONT']['intrinsic']}")
    else:
        print(f"❌ 找不到图片: {img_path}")

if __name__ == "__main__":
    test()
# 1. 构建镜像
docker build -t bev-engine-cpu .

# 2. 运行容器
# --shm-size=2gb: 关键！用于多进程共享内存，防止处理大图时崩溃
# -v "E:\code\P2:/app": 挂载代码
# -v "E:\code\nuscenes-mini:/data/nuscenes": 挂载数据集
docker run -it --rm `
    --name bev_dev_container `
    --shm-size=2gb `
    -v "E:\code\P2:/app" `
    -v "E:\code\nuscenes-mini:/data/nuscenes" `
    bev-engine-cpu
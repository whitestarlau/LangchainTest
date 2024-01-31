import torch

print(torch.__version__)

# 检查是否有可用的GPU
if torch.cuda.is_available():
    # 创建一个在GPU上的张量
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    print("PyTorch is using GPU for computation.")
else:
    print("PyTorch is using CPU for computation.")


if torch.cuda.is_available():
    # 获取当前GPU设备的CUDA版本
    cuda_version = torch.version.cuda
    print(f"CUDA 版本: {cuda_version}")
else:
    print("没有可用的GPU。")


import torch
import sys

print("=" * 50)
print("GPU DIAGNOSTIC")
print("=" * 50)
print(f"Python executable : {sys.executable}")
print(f"PyTorch version   : {torch.__version__}")

# 1. Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available    : {cuda_available}")

if cuda_available:
    # 2. Get GPU properties
    device_count = torch.cuda.device_count()
    print(f"GPU count          : {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}              : {props.name}")
        print(f"   VRAM Total      : {props.total_memory / 1024**3:.2f} GB")
        print(f"   Compute Cap     : {props.major}.{props.minor}")
    
    # 3. Test a small tensor operation on GPU
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.T)
        print(f"Test Tensor shape : {y.shape} (GPU working ✅)")
    except Exception as e:
        print(f"Test Tensor shape : FAILED ({e})")
else:
    print("❌ CUDA NOT DETECTED!")
    print("   If you are on Windows, try: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("   If you are on Linux, try: pip install torch --index-url https://download.pytorch.org/whl/cu124")

print("=" * 50)

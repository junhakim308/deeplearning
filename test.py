import torch

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("CUDA 디바이스 개수:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("사용 중인 디바이스 이름:", torch.cuda.get_device_name(0))
    print("디바이스 속도 (GPU):", torch.cuda.get_device_properties(0))
else:
    print("GPU 사용 불가, CPU로만 동작")

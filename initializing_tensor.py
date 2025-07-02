import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
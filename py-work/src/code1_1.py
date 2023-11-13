import torch

print("hello world!")

x = torch.tensor([1, 2, 3, 4])
print("CPU", x)

x = x.to("cuda")
print("GPU", x)

import torch

pos = torch.tensor([19,8])
grid = torch.tensor([20,10])

pos %= grid


print(pos)
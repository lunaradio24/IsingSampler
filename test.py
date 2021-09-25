import torch

a = torch.ones(1,2)
b = torch.ones(2,1)
c = torch.matmul(a,b)

print(c)
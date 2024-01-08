import torch
from torch.nn import functional as F
from torch import nn as nn

input = torch.randn(30, 20, requires_grad=True)
print(input)
target = torch.randint(21, (30,), dtype=torch.int64)
print(target)
loss = nn.CrossEntropyLoss()
loss1 = loss(input, target)
print(loss)
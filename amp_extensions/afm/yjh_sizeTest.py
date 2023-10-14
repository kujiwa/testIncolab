import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from csrc import _C
height=16
width =16
import torch

lines = torch.Tensor([[0, 0, 1, 1], [1, 4, 2, 3], [4, 4, 8, 7]])
print(lines)

print(_C)
lmap, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0))

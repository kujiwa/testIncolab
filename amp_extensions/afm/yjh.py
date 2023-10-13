import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from csrc import _C
height=255
width =255
lines=torch.empty((0,4))
print(_C)
lmap, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0))
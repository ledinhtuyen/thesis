from mmseg_custom import MAEAdapter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import math

while True:
    input = torch.randn(200, 3, 512, 512).cuda()

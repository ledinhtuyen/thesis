from model import mae_rvt_base_patch16
import torch

model = mae_rvt_base_patch16().to(0)
model(torch.randn(1, 3, 224, 224).to(0))

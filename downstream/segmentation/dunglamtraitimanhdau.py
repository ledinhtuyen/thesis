from mmseg_custom import MAEAdapter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = MAEAdapter(
  drop_path_rate=0.3, 
  init_values=1e-6, 
  deform_num_heads=16,
  deform_ratio=1.0, 
  with_cp=True,  # set with_cp=True to save memory
  interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
  use_vit_adapter=True,
  num_register_tokens=4,
).cuda()

import torch
import math

for i in range(999999999999999999):
  print(f"Epoch {i}")
  for j in range(725):
    print(f"Local step {j}")
    model.eval()
    output = model(torch.randn(16, 3, 352, 352).cuda())

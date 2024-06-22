import subprocess as sp
import os, sys

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

while True:
  memory_card = get_gpu_memory()
  if memory_card[1] >= 15000:
    print("GPU memory is enough to run pretrain")
    os.system("/bin/bash train_multitask.sh")
    break

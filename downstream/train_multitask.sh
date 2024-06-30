PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python train_multitask.py \
      --seed 140301 \
      --init_lr 1e-4 \
      --init_trainsize 384 \
      --warmup_epochs 1 \
      --num_epochs 20 \
      --batchsize 64 \
      --test_batchsize 64 \
      --accum_iter 1 \
      --metadata_file /home/s/tuyenld/endoscopy/multitask.json \
      --prefix_path /home/s/tuyenld/DATA \
      --config configs/mae/multitask/multitask.py \
      --work-dir work_dirs/multitask \
      --num_workers 16 \
      --amp

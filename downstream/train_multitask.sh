PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python train_multitask.py \
      --init_lr 5.0e-5 \
      --warmup_epochs 1 \
      --num_epochs 20 \
      --batchsize 2 \
      --test_batchsize 2 \
      --accum_iter 1 \
      --metadata_file /mnt/tuyenld/data/endoscopy/processed/multitask.json \
      --prefix_path /mnt/tuyenld/data/endoscopy/ \
      configs/mae/multitask.py \
      --work-dir work_dirs/multitask

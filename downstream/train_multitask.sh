PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_multitask.py \
      --init_lr 1e-4 \
      --init_trainsize 224 \
      --warmup_epochs 1 \
      --num_epochs 20 \
      --batchsize 64 \
      --test_batchsize 64 \
      --accum_iter 1 \
      --metadata_file /workspace/endoscopy/multitask.json \
      --prefix_path /workspace/DATA2 \
      configs/mae/multitask.py \
      --work-dir work_dirs/multitask \
      --num_workers 16 \
      --amp

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_multitask.py \
      --init_lr 1e-4 \
      --warmup_epochs 1 \
      --num_epochs 10 \
      --batchsize 200 \
      --test_batchsize 1200 \
      --accum_iter 1 \
      --metadata_file /workspace/endoscopy/multitask.json \
      --prefix_path /workspace/DATA/ \
      configs/mae/multitask.py \
      --work-dir work_dirs/multitask \
      --amp

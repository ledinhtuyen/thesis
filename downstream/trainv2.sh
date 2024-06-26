PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --init_lr 1e-4 \
      --warmup_epochs 1 \0
      --num_epochs 20 \
      --batchsize 4 \
      --test_batchsize 4 \
      --accum_iter 1 \
      --train_path /workspace/DATA2/public_dataset/TrainDataset \
      --work-dir work_dirs/benchmark/mae_continue_pretrain3 \
      --build_with_mmseg \
      --amp \
      --config configs/mae/rabithead/mae_base_meta.py

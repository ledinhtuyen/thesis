PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --seed 140301 \
      --init_lr 5e-5 \
      --warmup_epochs 1 \
      --num_epochs 100 \
      --batchsize 8 \
      --test_batchsize 8 \
      --accum_iter 4 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      --work-dir work_dirs/benchmark/mae_continue_pretrain4/testss \
      --amp \
      --config configs/mae/rabithead/mae_base_meta.py

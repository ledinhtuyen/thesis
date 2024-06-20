PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --init_lr 1.0e-4 \
      --warmup_epochs 2 \
      --num_epochs 100 \
      --batchsize 8 \
      --test_batchsize 8 \
      --accum_iter 8 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/mae_base_meta_rabithead.py \
      --work-dir work_dirs/mae_base_meta_rabithead \

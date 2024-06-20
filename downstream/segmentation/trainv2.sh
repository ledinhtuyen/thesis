PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --init_lr 5.0e-5 \
      --warmup_epochs 1 \
      --num_epochs 50 \
      --batchsize 4 \
      --test_batchsize 4 \
      --accum_iter 16 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/mae_base_meta_colonformerhead.py \
      --work-dir work_dirs/mae_base_meta_colonformerhead \

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --init_lr 1.0e-4 \
      --warmup_epochs 1 \
      --num_epochs 100 \
      --batchsize 4 \
      --test_batchsize 4 \
      --accum_iter 16 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/vit_adapter_mae_base_meta_rabithead.py \
      --work-dir work_dirs/vit_adapter_mae_base_meta_rabithead \

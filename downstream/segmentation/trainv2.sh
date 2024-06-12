PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --warmup_epochs 5 \
      --num_epochs 50 \
      --batchsize 2 \
      --test_batchsize 2 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/mae_base_meta_colonformerhead.py \
      --work-dir work_dirs/mae_base_meta_colonformerhead \
      --amp

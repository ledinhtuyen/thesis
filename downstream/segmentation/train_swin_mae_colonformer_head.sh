PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_swin_mae_colonformer_head.py \
      --init_trainsize 448 \
      --warmup_epochs 5 \
      --num_epochs 50 \
      --batchsize 8 \
      --test_batchsize 8 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/swin_mae_colonformerhead.py \
      --work-dir work_dirs/swin_mae_colonformerhead \

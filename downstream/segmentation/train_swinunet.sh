PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_swinunet.py \
      --init_trainsize 448 \
      --warmup_epochs 5 \
      --num_epochs 50 \
      --batchsize 8 \
      --test_batchsize 8 \
      --train_path /home/s/tuyenld/DATA/public_dataset/TrainDataset \
      configs/mae/swin_unet.py \
      --work-dir work_dirs/swin_unet \

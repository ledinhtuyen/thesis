PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train.py \
      --train.warmup_epochs 1 \
      --train.epochs 100 \
      --train.num_workers 8 \
      --train.num_gpus 1 \
      --train.lr_batch_size 1024 \
      --train.batch_size 128 \
      --train.lr 0.00005 \
      --train.dataset.type medical \
      --train.log_path /mnt/tuyenld/mae/pretrain/runs/pretrainv2/continue_pretrain_hiera_base_plus_224_medical_mae_v2 \
      --train.dataset.prefix_path /mnt/tuyenld/data/endoscopy \
      --train.dataset.path /mnt/tuyenld/data/endoscopy/processed/pretrain.json \
      --model hiera_base_plus_224 \
      --config medical_mae \
      --continue_pretrain

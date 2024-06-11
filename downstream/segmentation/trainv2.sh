PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python trainv2.py \
      --warmup_epochs 5 \
      --num_epochs 50 \
      --batchsize 4 \
      --test_batchsize 32 \
      --train_path ../../../data/endoscopy/public_dataset/TrainDataset/ \
      configs/mae/mae_base_adapter_v2.py \
      --work-dir work_dirs/mae_base_adapter_v2 \

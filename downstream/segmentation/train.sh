PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train.py \
      configs/mae/mae_base_adapter.py \
      --work-dir work_dirs/mae-base_adapter/upernet \
      --amp

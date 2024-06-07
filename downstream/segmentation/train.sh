PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python train.py \
      configs/mae/mae-base_dpt.py \
      --work-dir work_dirs/mae-base_dpt/exp2 \
      --amp

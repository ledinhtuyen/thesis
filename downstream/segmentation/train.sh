PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python train.py \
      configs/mae/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512.py \
      --work-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512 \
      --amp

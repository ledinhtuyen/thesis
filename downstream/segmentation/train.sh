CUDA_VISIBLE_DEVICES=0 \
python train.py \
      configs/mae/mae-base_upernet_8xb2-amp-40k_publicdataset-352x352.py \
      --work-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-352x352/exp1 \
      --amp

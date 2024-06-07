PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python test.py \
    configs/mae/mae-base_dpt.py \
    work_dirs/mae-base_dpt/exp1/iter_28000.pth \
    --work-dir work_dirs/mae-base_dpt/exp1/test_ETIS-LaribPolypDB \
    # --show \
    # --show-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp2/test_cvc_colondb/show_dir \
    # --out work_dirs/mae-base_upernet_8xb2-amp-80k_publicdataset-512x512_exp2/test_cvc_colondb/offline_eval \

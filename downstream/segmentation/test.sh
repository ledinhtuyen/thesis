PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python test.py \
    configs/mae/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512.py \
    work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp7/iter_24000.pth \
    --work-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp7/test_CVC-ClinicDB \
    # --show \
    # --show-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp2/test_cvc_colondb/show_dir \
    # --out work_dirs/mae-base_upernet_8xb2-amp-80k_publicdataset-512x512_exp2/test_cvc_colondb/offline_eval \

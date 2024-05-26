CUDA_VISIBLE_DEVICES=0 \
python test.py \
    configs/mae/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512.py \
    work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp3/iter_36000.pth \
    --work-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp3/test_CVC300 \
    # --show \
    # --show-dir work_dirs/mae-base_upernet_8xb2-amp-40k_publicdataset-512x512_exp2/test_cvc_colondb/show_dir \
    # --out work_dirs/mae-base_upernet_8xb2-amp-80k_publicdataset-512x512_exp2/test_cvc_colondb/offline_eval \

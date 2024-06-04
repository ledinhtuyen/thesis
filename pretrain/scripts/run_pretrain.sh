CUDA_VISIBLE_DEVICES=1 python run_pretraining.py \
                       --cfg pretrain \
                       --exp_name="mae_vit_large_patch16_with_register_v2" \
                       --norm_pix_loss

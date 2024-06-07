CUDA_VISIBLE_DEVICES=1 python run_pretraining.py \
                       --cfg pretrain \
                       --exp_name="mae_vit_base_patch16_with_register_convstem_normpix" \
                       --norm_pix_loss

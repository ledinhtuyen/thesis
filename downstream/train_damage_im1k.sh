PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_damage.py \
      --seed 140301 \
      --init_lr 5e-5 \
      --warmup_epochs 1 \
      --num_epochs 20 \
      --batchsize 16 \
      --test_batchsize 16 \
      --accum_iter 1 \
      --type_damage ung_thu_da_day_20230620 \
      --num_workers 8 \
      --prefix_path /home/s/tuyenld/DATA \
      configs/mae/mae_base_meta_im1k_rabithead.py \
      --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage/ \
      --amp
# PYTHONDONTWRITEBYTECODE=1 \
# CUDA_VISIBLE_DEVICES=0 \
# python train_damage.py \
#       --init_lr 1.0e-4 \
#       --warmup_epochs 1 \
#       --num_epochs 20 \
#       --batchsize 8 \
#       --test_batchsize 8 \
#       --accum_iter 4 \
#       --type_damage ung_thu_thuc_quan_20230620 \
#       --num_workers 8 \
#       --prefix_path /home/s/tuyenld/DATA \
#       configs/mae/mae_base_meta_im1k_rabithead.py \
#       --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage \
#       --amp
# PYTHONDONTWRITEBYTECODE=1 \
# CUDA_VISIBLE_DEVICES=0 \
# python train_damage.py \
#       --init_lr 1.0e-4 \
#       --warmup_epochs 1 \
#       --num_epochs 20 \
#       --batchsize 8 \
#       --test_batchsize 8 \
#       --accum_iter 4 \
#       --type_damage viem_da_day_20230620 \
#       --num_workers 8 \
#       --prefix_path /home/s/tuyenld/DATA \
#       configs/mae/mae_base_meta_im1k_rabithead.py \
#       --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage \
#       --amp
# PYTHONDONTWRITEBYTECODE=1 \
# CUDA_VISIBLE_DEVICES=0 \
# python train_damage.py \
#       --init_lr 1.0e-4 \
#       --warmup_epochs 1 \
#       --num_epochs 20 \
#       --batchsize 8 \
#       --test_batchsize 8 \
#       --accum_iter 4 \
#       --type_damage viem_thuc_quan_20230620 \
#       --num_workers 8 \
#       --prefix_path /home/s/tuyenld/DATA \
#       configs/mae/mae_base_meta_im1k_rabithead.py \
#       --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage \
#       --amp
# PYTHONDONTWRITEBYTECODE=1 \
# CUDA_VISIBLE_DEVICES=0 \
# python train_damage.py \
#       --init_lr 1.0e-4 \
#       --warmup_epochs 1 \
#       --num_epochs 20 \
#       --batchsize 8 \
#       --test_batchsize 8 \
#       --accum_iter 4 \
#       --type_damage viem_loet_hoanh_ta_trang_20230620 \
#       --num_workers 8 \
#       --prefix_path /home/s/tuyenld/DATA \
#       configs/mae/mae_base_meta_im1k_rabithead.py \
#       --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage \
#       --amp
# PYTHONDONTWRITEBYTECODE=1 \
# CUDA_VISIBLE_DEVICES=0 \
# python train_damage.py \
#       --init_lr 1.0e-4 \
#       --warmup_epochs 1 \
#       --num_epochs 20 \
#       --batchsize 8 \
#       --test_batchsize 8 \
#       --accum_iter 4 \
#       --type_damage polyp \
#       --num_workers 8 \
#       --prefix_path /home/s/tuyenld/DATA \
#       configs/mae/mae_base_meta_im1k_rabithead.py \
#       --work-dir work_dirs/mae_base_meta_im1k_rabithead_damage \
#       --amp

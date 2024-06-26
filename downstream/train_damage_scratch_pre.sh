PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python train_damage.py \
      --seed 140301 \
      --init_lr 1e-4 \
      --init_trainsize 384 \
      --warmup_epochs 1 \
      --num_epochs 20 \
      --batchsize 8 \
      --test_batchsize 8 \
      --accum_iter 1 \
      --type_damage viem_thuc_quan_20230620 \
      --num_workers 8 \
      --prefix_path /workspace/DATA2 \
      --work-dir work_dirs/damage/rabithead/mae_scratch \
      --amp \
      --build_with_mmseg \
      --config configs/mae/rabithead/mae_base_meta_scratch_pre.py

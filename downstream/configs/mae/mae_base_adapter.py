_base_ = [
    '../_base_/datasets/public_dataset.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(imports=["mmseg_custom"])

pretrained = "../../pretrain/runs/pretrainv2/mae_meta_register_norm_pix_img224_p16/weight/last.pth"

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(352, 352),
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MAEAdapter',
        drop_path_rate=0.3, 
        init_values=1e-6, 
        deform_num_heads=16,
        deform_ratio=1.0, 
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        use_vit_adapter=True,
        num_register_tokens=4,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1, # 1 for binary classification
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            dict(type="StructureLoss", loss_type="focal", loss_weight=1.0)
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1, # 1 for binary classification
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            dict(type="StructureLoss", loss_type="focal", loss_weight=1.0)
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'pos_embed': dict(decay_mult=0.),
    #         'cls_token': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         'register_tokens': dict(decay_mult=0.),
    #     }),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    constructor='LayerDecayOptimizerConstructor_Adapter',
    # accumulative_counts=16
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    pin_memory=True,
)
val_dataloader = dict(
    batch_size=32,
    num_workers=16,
    pin_memory=True,
)
test_dataloader = val_dataloader

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[117.9630,  79.7130,  63.8520],
    std=[81.4725, 63.9030, 56.2275],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit-b16_p16_224-80ecf9dd.pth', # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
        final_norm=False,
        output_cls_token=False),
    decode_head=dict(
        type='UNETRHead',
        embed_dims=768,
        in_channels=3,
        channels=256,
        num_classes=150,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable

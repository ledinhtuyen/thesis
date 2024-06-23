pretrained = "/home/s/tuyenld/mae/pretrain/runs/pretrainv2/vit-base-mae/mae_pretrain_vit_base.pth"
model = dict(
    type='EncoderDecoderCustom',
    pretrained=pretrained,
    backbone=dict(
        type='MaskedAutoencoderViT',
        downstream_size=384,
        num_register_tokens=0,
        out_indices=(-1, 2, 5, 8, 11),
    ),
    neck=None,
    decode_head=dict(
        type='UNETRHead',
        embed_dims=768,
        in_channels=3,
        channels=256,
        num_classes=2,
        out_channels=1,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

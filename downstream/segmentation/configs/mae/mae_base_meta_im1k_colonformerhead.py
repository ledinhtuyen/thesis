pretrained = "/mnt/tuyenld/mae/pretrain/runs/pretrainv2/vit-mae-base/mae_pretrain_vit_base.pth"
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderV2',
    pretrained=pretrained,
    backbone=dict(
        type='MaskedAutoencoderViT',
        downstream_size=352,
        num_register_tokens=0,
        out_indices=(3, 5, 7, 11),
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1, # 1 for binary classification
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

test_path = dict(
    Kvasir="/home/s/tuyenld/DATA/public_dataset/TestDataset/Kvasir",
    CVC_ClinicDB="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB",
    CVC_ColonDB="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ColonDB",
    CVC_T="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-300",
    ETIS_Larib="/home/s/tuyenld/DATA/public_dataset/TestDataset/ETIS-LaribPolypDB",
)

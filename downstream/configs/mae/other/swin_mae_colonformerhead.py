pretrained = "/home/s/tuyenld/Swin-MAE/output_dir/checkpoint-400.pth"
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderCustom',
    pretrained=pretrained,
    backbone=dict(
        type='swin_mae',
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
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

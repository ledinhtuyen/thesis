pretrained = "../../pretrain/runs/pretrainv2/mae_vit_large_patch16_with_register_v2/weight/last.pth"
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderV2',
    pretrained=pretrained,
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
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
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
    Kvasir="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/Kvasir",
    CVC_ClinicDB="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-ClinicDB",
    CVC_ColonDB="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-ColonDB",
    CVC_T="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-300",
    ETIS_Larib="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/ETIS-LaribPolypDB",
)

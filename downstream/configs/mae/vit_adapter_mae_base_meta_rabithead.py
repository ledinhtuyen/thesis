pretrained = "/home/s/tuyenld/mae/pretrain/runs/pretrainv2/continue_pretrain/weight/epoch_50.pth"
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderRaBiT',
    pretrained=pretrained,
    backbone=dict(
        type='MAEAdapter',
        drop_path_rate=0.3, 
        init_values=1e-6, 
        deform_num_heads=16,
        deform_ratio=1.0, 
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        num_register_tokens=0,
    ),
    # neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=None,
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

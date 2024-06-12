pretrained = "/home/s/tuyenld/Swin-MAE/output_dir/checkpoint-400.pth"
model = dict(
    type='SwinUnet',
    pretrained=pretrained,
    num_classes=1,
    window_size=7,
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

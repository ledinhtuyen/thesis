# dataset settings
dataset_type = 'PublicDataset'
img_scale = (512, 512)

albu_transforms = [
    dict(type='ShiftScaleRotate', p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
    dict(type='RandomGamma', p=0.5),
    dict(type='RGBShift', p=0.3, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
    dict(type='OneOf', transforms=[
        dict(type='Blur'),
        dict(type='GlassBlur'),
        dict(type='GaussianBlur'),
        dict(type='GaussNoise'),
        dict(type='MotionBlur'),
        dict(type='Sharpen'),
        dict(type='MedianBlur'),
        dict(type='MultiplicativeNoise'),
    ], p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion'),
    dict(type='Albu', transforms=albu_transforms),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file="../data/train_public_dataset.txt",
        data_prefix=dict(img_path="/home/s/tuyenld/DATA/public_dataset/TrainDataset/image", 
                         seg_map_path="/home/s/tuyenld/DATA/public_dataset/TrainDataset/masks"),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            ann_file="../data/test_CVC-ClinicDB.txt",
            img_path='/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB/images',
            seg_map_path='/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB/masks'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

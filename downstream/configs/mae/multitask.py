pretrained = "/workspace/mae_continue_pretrain/epoch_90.pth"
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='MultiTask',
    pretrained=pretrained,
    backbone=dict(
        type='MaskedAutoencoderViT',
        downstream_size=384,
        num_register_tokens=0,
        out_indices=(2, 5, 8, 11),
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

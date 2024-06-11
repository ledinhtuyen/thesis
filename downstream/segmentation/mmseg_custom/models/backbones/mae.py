# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.layers import PatchEmbedWithSize

# --------------------------------------------------------
# Position embedding
# --------------------------------------------------------

import numpy as np
import math


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=strides[i],
                    padding=1,
                    bias=(not batch_norm),
                )
            ]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i + 1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [
            nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])
        ]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod) ** 2
        self.patch_size = (stride_prod, stride_prod)

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


from mmseg.registry import MODELS

@MODELS.register_module()
class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_register_tokens=4,
        out_indices=-1,
        init_cfg=None,
        final_norm=True,
        interpolate_mode="bicubic",
        use_conv_stem=False,
        output_cls_token=False,
        use_vit_adapter=False,
        **kwargs,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        self.final_norm = final_norm
        self.interpolate_mode = interpolate_mode
        self.use_conv_stem = use_conv_stem
        self.output_cls_token = output_cls_token
        self.use_vit_adapter = use_vit_adapter

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if self.use_conv_stem:
            self.patch_embed = ConvEmbed(
                channels=[48, 96, 192, 384, embed_dim],
                strides=[2, 2, 2, 2, 1],
                img_size=(img_size, img_size),
                batch_norm=False,
            )
        else:
            self.patch_embed = PatchEmbedWithSize(
                img_size, patch_size, in_chans, embed_dim
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        assert num_register_tokens >= 0
        if num_register_tokens > 0:
            print(f"Using {num_register_tokens} register tokens")
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim)
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if self.use_conv_stem:
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            torch.nn.init.normal_(self.register_tokens, std=1e-6)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        self.apply(self._init_weights)

        if (
            isinstance(self.init_cfg, dict) 
            and self.init_cfg.get("type") == "Pretrained"
        ) or pretrained is not None:
            from mmengine.runner.checkpoint import _load_checkpoint

            if pretrained is not None:
                checkpoint = _load_checkpoint(
                    pretrained, logger=None, map_location="cpu"
                )
            else:
                checkpoint = _load_checkpoint(
                    self.init_cfg["checkpoint"], logger=None, map_location="cpu"
                )
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            if "pos_embed" in state_dict.keys() and self.use_vit_adapter == False:
                if self.pos_embed.shape != state_dict["pos_embed"].shape:
                    from mmengine.logging import print_log

                    print_log(
                        msg=f"Resize the pos_embed shape from "
                        f'{state_dict["pos_embed"].shape} to '
                        f"{self.pos_embed.shape}"
                    )
                    h, w = (self.img_size, self.img_size)
                    pos_size = int(math.sqrt(state_dict["pos_embed"].shape[1] - 1))
                    state_dict["pos_embed"] = self.resize_pos_embed(
                        state_dict["pos_embed"],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size),
                        self.interpolate_mode,
                    )
            self.load_state_dict(state_dict, False)
        else:
            self.initialize_weights()

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)

        from mmseg.models.utils import resize

        pos_embed_weight = resize(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode
        )
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward_encoder(self, x):
        outs = []

        if -1 in self.out_indices:
            outs.append(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if self.register_tokens is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = torch.cat(
                (
                    cls_tokens,
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)

            if i in self.out_indices:
                out = x[:, 1 + self.num_register_tokens :]
                B, _, C = out.shape

                out = (
                    out.reshape(B, self.grid_size, self.grid_size, C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def forward(self, imgs):
        return self.forward_encoder(imgs)

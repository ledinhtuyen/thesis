from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed

from model import RvT


class MaskedAutoencoderRvT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = RvT(image_size=img_size, patch_size=patch_size, dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=embed_dim * mlp_ratio, channels=in_chans, use_ds_conv=False)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder = RvT(image_size=img_size, patch_size=patch_size, dim=decoder_embed_dim, depth=decoder_depth, heads=decoder_num_heads, mlp_dim=decoder_embed_dim * mlp_ratio, channels=in_chans, is_decoder=True, use_ds_conv=False)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x, h, w, mask_ratio):
        x = self.encoder.to_patch_embedding(x)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # apply RvT
        x = self.encoder(x, h, w)

        return x, mask, ids_restore

    def forward_decoder(self, x, h, w, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply RvT
        x = self.decoder(x, h, w)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        _, _, H, W = imgs.shape
        latent, mask, ids_restore = self.forward_encoder(imgs, H, W, mask_ratio)
        pred = self.forward_decoder(latent, H, W, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_rvt_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderRvT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_rvt_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderRvT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_rvt_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderRvT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_rvt_base_patch32_384_dec512d8b(**kwargs):
    model = MaskedAutoencoderRvT(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_rvt_base_patch16 = mae_rvt_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_rvt_base_patch32_384 = mae_rvt_base_patch32_384_dec512d8b
mae_rvt_large_patch16 = mae_rvt_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_rvt_huge_patch14 = mae_rvt_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

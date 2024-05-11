import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import warnings
warnings.filterwarnings("ignore")

class Transformer(nn.Module):
    """Transformer Encoder 
    Args:
        embedding_dim: dimension of embedding
        n_heads: number of attention heads
        n_layers: number of attention layers
        feedforward_dim: hidden dimension of MLP layer
    Returns:
        Transformer embedding of input
    """
    def __init__(self, embedding_dim=768, n_heads=12, n_layers=12, feedforward_dim=3072, p_dropout=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=self.n_heads,
                dim_feedforward=self.feedforward_dim,
                activation=F.gelu,
                batch_first=True,
                dropout=p_dropout,
                norm_first=True
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim, eps=1e-6)
        )

    def forward(self, x):
        return self.transformer(x)
    
class MaskedAutoEncoder(nn.Module):
    """MAE Encoder
    Args:
        encoder: vit encoder
        decoder: vit decoder
        encoder_embedding_dim: embedding size of encoder
        decoder_embedding_dim: embedding size of decoder
        patch_size: image patch size
        num_patches: number of patches
        mask_ratio: percentage of masked patches
    """
    def __init__(self, img_size=224, 
                  patch_size=16, 
                  encoder_embedding_dim=768, 
                  encoder_layers=12,
                  n_heads_encoder_layer=12,
                  decoder_embedding_dim=512,
                  decoder_layers=8,
                  n_heads_decoder_layer=16,
                  p_dropout=0.5,
                  mask_ratio=0.75,
                  num_register_tokens=0,
                  norm_pix_loss=False):
        super().__init__()
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.image_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.num_register_tokens = num_register_tokens

        self.masked_length = int(self.num_patches * mask_ratio)
        self.keep_length = self.num_patches - self.masked_length

        self.encoder = Transformer(
            embedding_dim=encoder_embedding_dim,
            n_layers=encoder_layers,
            n_heads=n_heads_encoder_layer,
            feedforward_dim=encoder_embedding_dim * 4,
            p_dropout=p_dropout
        )
        
        self.decoder = Transformer(
            embedding_dim=decoder_embedding_dim,
            n_layers=decoder_layers,
            n_heads=n_heads_decoder_layer,
            feedforward_dim=decoder_embedding_dim * 4,
            p_dropout=p_dropout
        )

        self.encoder_input_projection = nn.Linear(patch_size * patch_size * 3, encoder_embedding_dim)
        self.decoder_input_projection = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.decoder_output_projection = nn.Linear(decoder_embedding_dim, patch_size * patch_size * 3)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim))
        self.encoder_position_encoding = nn.Parameter(torch.zeros(1, self.num_patches, encoder_embedding_dim))
        self.decoder_position_encoding = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embedding_dim))
        self.masked_tokens = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, encoder_embedding_dim)) if num_register_tokens else None
        )

        self.norm_pix_loss = norm_pix_loss
        
        self.init_weights()
        
    def init_weights(self):
        trunc_normal_(self.encoder_position_encoding, std=0.02)
        trunc_normal_(self.decoder_position_encoding, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        # named_apply(init_weights_vit_timm, self)

    def patchify(self, imgs):
        """Splitting images into patches.
        Args:
            imgs: Input tensor with size (batch, channels, height, width)
                We can assume that image is square where height == width.
        Returns:
            A batch of image patches with size (
              batch, (height / patch_size) * (width / patch_size), 
            channels * patch_size * patch_size)
        """
        # BEGIN CODE
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
        # END CODE

    def unpatchify(self, x):
        """Combining patches into images.
        Args:
            x: Input tensor with size (
            batch, (height / patch_size) * (width / patch_size), 
            channels * patch_size * patch_size)
        Returns:
            A batch of images with size (batch, channels, height, width)
        """
        # BEGIN CODE
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
        # END CODE

    def index_sequence(self, x, ids):
        """Index tensor (x) with indices given by ids
        Args:
            x: input sequence tensor, can be 2D (batch x length) or 3D (batch x length x feature)
            ids: 2D indices (batch x length) for re-indexing the sequence tensor
        """
        if len(x.shape) == 3:
            ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.take_along_dim(x, ids, dim=1)

    def random_masking(self, x, keep_length, ids_shuffle):
        """Apply random masking on input tensor
        Args:
            x: input patches (batch x length x feature)
            keep_length: length of unmasked patches
            ids_shuffle: random indices for shuffling the input sequence 
        Returns:
            kept: unmasked part of x
            mask: a 2D (batch x length) mask tensor of 0s and 1s indicated which
                part of x is masked out. The value 0 indicates not masked and 1
                indicates masked.
            ids_restore: indices to restore x. If we take the kept part and masked
                part of x, concatentate them together and index it with ids_restore,
                we should get x back.

        Note:
            ids_shuffle contains the indices used to shuffle the sequence (patches).
            Use the provided index_sequence function to re-index the
            sequence, and keep the first keep_length number of patches.
        """
        # BEGIN CODE
        b, l, f = x.shape

        ids_restore = torch.argsort(ids_shuffle, dim = 1)
        kept = self.index_sequence(x, ids_shuffle)
        kept = kept[:, :keep_length]
        ids_kept = ids_shuffle[:, :keep_length]
        
        mask = torch.ones([b, l], device=x.device)
        mask[:, :keep_length] = 0
        mask = torch.gather(mask, dim=1, index = ids_restore)
        return kept, mask, ids_restore
        # END CODE

    def restore_masked(self, kept_x, masked_x, ids_restore):
        """Restore masked patches
        Args:
            kept_x: unmasked patches
            masked_x: masked patches
            ids_restore: indices to restore x
        Returns:
            restored patches
        """
        # BEGIN CODE
        x = torch.cat((kept_x, masked_x), dim = 1)
        x = torch.gather(x, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        return x
        
        # END CODE
    def forward_encoder(self, images, ids_shuffle=None):
        """Encode input images
        You should implement the following steps
        (1) patchify images into patches
        (2) linear projection
        (3) add position encoding
        (4) concatenate cls_token and patches embedding and pass it to vit encoder
        """
        batch_size = images.shape[0]
        # Generate random shuffling indices
        if ids_shuffle is None:
            ids_shuffle = torch.argsort(
                torch.rand(
                    (batch_size, self.num_patches),
                    device=images.device
                ),
                dim=1
            )
        # BEGIN CODE
        #1
        patches = self.patchify(images)

        #2
        projection = self.encoder_input_projection(patches)

        #3
        b, n, l = projection.shape
        projection += self.encoder_position_encoding[:, :(n + 1)]

        #4
        kept, mask, ids_restore = self.random_masking(projection, n - self.masked_length, ids_shuffle)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, kept), dim=1)
        encoder_output = self.encoder(x)
        
        return encoder_output, mask, ids_restore
        # END CODE

    def forward_decoder(self, encoder_embeddings, ids_restore):
        """Decode encoder embeddings
        You should implement the following steps
        (1) linear projection of encoder embeddings
        (2) restore sequence from masked_patches and encoder predictions
        (3) add position encoding
        (3) readd/use CLS token and decode using ViT decoder 
        (4) projection to predict image patches
        """
        # BEGIN CODE
        
        #1
        projection = self.decoder_input_projection(encoder_embeddings)
        b, n, l = projection.shape
        #2
        cls_tokens = projection[:, :1, :]
        embedding = projection[:, 1:, :]
        mask_tokens = self.masked_tokens.repeat(b, ids_restore.shape[1] - n + 1, 1)
        
        x = self.restore_masked(embedding, mask_tokens, ids_restore)
        
        #3
        b, n, l = x.shape
        x += self.decoder_position_encoding[:, :(n + 1)]
        
        #4
        x = torch.cat((cls_tokens, x), dim = 1)
        
        decoder_output = self.decoder(x)
        decoder_output = decoder_output[:, 1:, :]
        #5
        return self.decoder_output_projection(decoder_output)
        # END CODE
    
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, images):
        encoder_output, mask, ids_restore = self.forward_encoder(images)
        decoder_output = self.forward_decoder(encoder_output, ids_restore)
        loss = self.forward_loss(images, decoder_output, mask)
        return loss, decoder_output, mask

    def forward_encoder_representation(self, images):
        """Encode images without applying random masking to get representation
        of input images. 

        Implement splitting images into patches, readd/use CLS token,
        and encoding with ViT encoder.
        """
        # BEGIN CODE
        patches = self.patchify(images)

        #2
        projection = self.encoder_input_projection(patches)

        #3
        b, n, l = projection.shape
        projection += self.encoder_position_encoding[:, :(n + 1)]

        #4
        cls_tokens = einops.repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, projection), dim=1)
        encoder_output = self.encoder(x)
        return encoder_output
        # END CODE

def mae_base_by_tuyenld_patch16_dec512d8b(**kwargs):
  return MaskedAutoEncoder(
    patch_size=16,
    encoder_embedding_dim=768, encoder_layers=12, n_heads_encoder_layer=12,
    decoder_embedding_dim=512, decoder_layers=8, n_heads_decoder_layer=16,
    p_dropout=0.1,
    **kwargs
  )
  
def mae_base_by_tuyenld_patch32_dec512d8b(**kwargs):
  return MaskedAutoEncoder(
    patch_size=32,
    encoder_embedding_dim=768, encoder_layers=12, n_heads_encoder_layer=12,
    decoder_embedding_dim=512, decoder_layers=8, n_heads_decoder_layer=16,
    p_dropout=0.1,
    **kwargs
  )

def mae_large_by_tuyenld_patch16_dec512d8b(**kwargs):
  return MaskedAutoEncoder(
    patch_size=16,
    encoder_embedding_dim=1024, encoder_layers=24, n_heads_encoder_layer=16,
    decoder_embedding_dim=512, decoder_layers=8, n_heads_decoder_layer=16,
    p_dropout=0.1,
    **kwargs
  )
  
def mae_large_by_tuyenld_patch32_dec512d8b(**kwargs):
  return MaskedAutoEncoder(
    patch_size=32,
    encoder_embedding_dim=1024, encoder_layers=24, n_heads_encoder_layer=16,
    decoder_embedding_dim=512, decoder_layers=8, n_heads_decoder_layer=16,
    p_dropout=0.1,
    **kwargs
  )

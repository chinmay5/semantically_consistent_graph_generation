# This source code is licensed under the license found in the
# original SiT project.
# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT/blob/main/models.py
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = x * emb[None, :]
        emb = x.unsqueeze(-1) * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1).mean(2)
        return emb



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AttentionWithPads(Attention):
    def __init__(self, dim, num_heads, qkv_bias, **block_kwargs):
        super(AttentionWithPads, self).__init__(dim=dim, num_heads=num_heads,
                                                qkv_bias=qkv_bias, **block_kwargs)
        # Nothing else is needed here. It is the forward function that will be changed

    def forward(self, x, padding_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Now, we shall apply the mask here
        single_head_mask = (padding_mask.unsqueeze(1) & padding_mask.unsqueeze(-1))  # B, N, N
        multi_head_mask = single_head_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn = attn.masked_fill(multi_head_mask == 0, -1e9)  # putting -inf will cause nan for all-masked rows.
        # Same is done in pytorch -> https://github.com/pytorch/pytorch/issues/41508#issuecomment-1431914890
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithPads(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, padding_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=1152,
        num_transformer_blocks=28,
        num_heads=16,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        # translate each token from in_channels into hidden_size
        self.pos_embedder = torch.nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.ff_encoder = SinusoidalPosEmb(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_transformer_blocks)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # nn.init.xavier_uniform_(self.x_embedder.weight)
        # nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.pos_embedder.weight)
        nn.init.constant_(self.pos_embedder.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, padding_mask):
        """
        Forward pass of SiT.
        x: (B, N, H) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        pos: (B, N, 3) tensor of raw coordinate positions
        """
        x = self.pos_embedder(x) # (N, T, D)
        # FF only for the node coordinates
        # ff = self.ff_encoder(x)
        # ff[:, :, 3:] = 0
        # x = x + ff
        t = self.t_embedder(t)                     # (N, D)
        c = t
        for block in self.blocks:
            x = block(x, c, padding_mask)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, out_channels)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL(**kwargs):
    return SiT(num_transformer_blocks=28, hidden_size=1152, num_heads=16, **kwargs)


def SiT_L(**kwargs):
    return SiT(num_transformer_blocks=24, hidden_size=1024, num_heads=16, **kwargs)


def SiT_B(**kwargs):
    return SiT(num_transformer_blocks=12, hidden_size=768, num_heads=12, **kwargs)


def SiT_S(**kwargs):
    return SiT(num_transformer_blocks=12, hidden_size=384, num_heads=6, **kwargs)

def SiT_T(**kwargs):
    return SiT(num_transformer_blocks=6, hidden_size=192, num_heads=6, **kwargs)

def SiT_VT(**kwargs):
    return SiT(num_transformer_blocks=4, hidden_size=192, num_heads=4, **kwargs)


class SiT_Clf(nn.Module):
    def __init__(
            self,
            in_channels=3,
            hidden_size=128,
            num_transformer_blocks=4,
            num_heads=4,
            mlp_ratio=4.0,
            # class_dropout_prob=0.1,
            out_channels=1,
    ):
        super(SiT_Clf, self).__init__()
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_transformer_blocks)
        ])
        self.pos_embedder = torch.nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.clf = nn.Linear(hidden_size, out_channels)

    def forward(self, x, padding_mask):
        t = torch.ones(x.shape[0], device=x.device)
        x = self.pos_embedder(x)
        t = self.t_embedder(t)  # (N, D)
        for block in self.blocks:
            x = block(x, t, padding_mask)
        # Do a mean pooling over all the points
        x = x.mean(dim=1)
        return self.clf(x)



SiT_models = {
    'SiT-XL': SiT_XL,
    'SiT-L':  SiT_L,
    'SiT-B':  SiT_B,
    'SiT-S':  SiT_S,
    'SiT-T':  SiT_T,
    'SiT-VT':  SiT_VT,

}

if __name__ == '__main__':
    model = SiT_models['SiT-S'](in_channels=17)
    pos = torch.randn(2, 8, 17)
    x = torch.randn(2, 8, 16)
    t = torch.rand(2,)
    padding_mask = (torch.rand(2, 8) > 0.5)
    print(model(pos, t, padding_mask).shape)
    clf = SiT_Clf(in_channels=17)
    print(clf(pos, padding_mask).shape)
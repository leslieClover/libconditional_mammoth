"""
Vision Transformer with prompt integration for STAR-Prompt.
Based on the open-source implementation with proper ViT architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class Attention(nn.Module):
    """
    Attention layer as used in Vision Transformer.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to q, k, v
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate after the final projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        """
        Forward pass of the attention layer.

        Args:
            x: Input tensor
        """

        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # NOTE: flash attention is less debuggable than the original. Use the commented code below if in trouble.
        # check torch version
        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            warn_once("Torch verison < 2.1.0 detected. Using the original attention code.")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ResidualPromptAttention(Attention):
    """Multi-head attention with residual prompt support"""
    # def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    #     super().__init__()
    #     assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    #     self.num_heads = num_heads
    #     head_dim = dim // num_heads
    #     self.scale = head_dim ** -0.5

    #     self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    #     self.attn_drop = nn.Dropout(attn_drop)
    #     self.proj = nn.Linear(dim, dim)
    #     self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompts=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if prompts is not None:
            prompts = prompts.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + prompts

        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrefixTuningAttention(Attention):
    """Multi-head attention with prefix tuning support"""
    # def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    #     super().__init__()
    #     assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    #     self.num_heads = num_heads
    #     head_dim = dim // num_heads
    #     self.scale = head_dim ** -0.5

    #     self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    #     self.attn_drop = nn.Dropout(attn_drop)
    #     self.proj = nn.Linear(dim, dim)
    #     self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompts=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if prompts is not None:
            prompt_k, prompt_v = prompts
            # Reshape prompts
            prompt_k = prompt_k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            prompt_v = prompt_v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # Concatenate prompts
            k = torch.cat([prompt_k, k], dim=2)
            v = torch.cat([prompt_v, v], dim=2)

        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """Transformer block with prompt support"""
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4.,
                 qkv_bias=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 init_values=None, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 attn_layer=ResidualPromptAttention):
        super().__init__()
        self.embed_dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prompts=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompts)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with prompt integration for STAR-Prompt"""

    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 init_values=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=None, 
                 act_layer=None, 
                 prompt_mode='residual', 
                 clip_embed_dim=512, 
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.prompt_mode = prompt_mode
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Choose attention layer based on prompt mode
        attn_layer = ResidualPromptAttention if prompt_mode == 'residual' else PrefixTuningAttention

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], init_values=init_values,
                norm_layer=norm_layer, act_layer=act_layer, attn_layer=attn_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize patch embedding
        if hasattr(self.patch_embed, 'proj'):
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize head
        if isinstance(self.head, nn.Linear):
            torch.nn.init.trunc_normal_(self.head.weight, std=.02)
            torch.nn.init.zeros_(self.head.bias)
        
        # Apply weight initialization to all modules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)

    def _pos_embed(self, x):
        """Add positional embedding"""
        return x + self.pos_embed

    def forward_features(self, x, first_stage_query, prompter, cur_classes: int, frozen_past_classes=0):
        """Forward pass for feature extraction with prompts"""
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = self._pos_embed(x)
        x = self.pos_drop(x)
        
        # Pass through transformer blocks with prompts
        for idx, blk in enumerate(self.blocks):
            prompts = prompter.get_prompts(idx, first_stage_query, cur_classes=cur_classes, 
                                         frozen_past_classes=frozen_past_classes)
            x = blk(x, prompts)
        
        x = self.norm(x)
        return x

    def forward_head(self, features):
        """Forward pass for classification head"""
        # Use cls token for classification
        cls_features = features[:, 0]
        return self.head(cls_features)

    def forward(self, x: torch.Tensor, first_stage_query: torch.Tensor, prompter, cur_classes: int, 
                frozen_past_classes=0) -> torch.Tensor:
        """
        Complete forward pass with prompt integration.

        Args:
            x: input image
            first_stage_query: the output of the visual encoder of CLIP, to be used as query for the second stage's prompter
            prompter: the prompter of the second stage
            cur_classes: number of current classes
            frozen_past_classes: number of frozen past classes
        """
        x = self.forward_features(x, first_stage_query, prompter, cur_classes, frozen_past_classes)
        x = self.forward_head(x)
        return x
"""
Second stage model for STAR-Prompt.
Implements Vision Transformer with prompting for final classification.
"""

import logging
import torch
import torch.nn as nn
from typing import List

from .vision_transformer import VisionTransformer
from .second_stage_prompter import Prompter

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")


class SecondStageModel(nn.Module):
    """Second stage model for STAR-Prompt"""

    def __init__(self, args, backbone: nn.Module, num_classes, device='cpu',
                 clip_model: clip.model.CLIP = None, clip_preprocess=None):
        super().__init__()

        assert 'resnet' not in str(type(backbone)).lower(), "ResNet not supported"

        self.args = args
        self.num_classes = num_classes
        self.device = device

        # Get CLIP embedding dimension
        if clip_model is None:
            # Load CLIP temporarily to get embedding dimension
            clip_backbone = args.clip_backbone if hasattr(args, 'clip_backbone') else 'ViT-B/32'
            temp_clip, _ = clip.load(clip_backbone, device)
            clip_embed_dim = temp_clip.visual.output_dim
            del temp_clip
        else:
            clip_embed_dim = clip_model.visual.output_dim



        # Initialize Vision Transformer
        # vit_model = VisionTransformer(
        #     img_size=224,
        #     patch_size=16,
        #     in_chans=3,
        #     num_classes=num_classes,
        #     embed_dim=768,
        #     depth=12,
        #     num_heads=12,
        #     mlp_ratio=4.,
        #     qkv_bias=True,
        #     drop_rate=0.1,
        #     attn_drop_rate=0.1,
        #     drop_path_rate=0.0,
        #     prompt_mode=args.prompt_mode,
        #     clip_embed_dim=clip_embed_dim
        # ).to(device)
        vit_model = VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0,
            num_classes=num_classes,
            prompt_mode=args.prompt_mod
        ).to(device)

        # logging.info("Loading the Vision Transformer backbone...")
        # load_dict = backbone.state_dict()
        # for k in list(load_dict.keys()):
        #     if 'head' in k:
        #         del load_dict[k]
        # missing, unexpected = vit_model.load_state_dict(load_dict, strict=False)
        # assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
        # assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

        self.vit = vit_model

        # Set up prompt layers - use the actual depth of the ViT
        # self.prompt_layers = list(range(vit_depth))
        self.prompt_layers = list(range(len(self.vit.blocks)))

        logging.info("Initializing the prompter and prompt parameters...")
        self.prompter = Prompter(
            args,
            num_classes=num_classes,
            target_embed_len=self.vit.patch_embed.num_patches,  # 14*14 patches for 224x224 image with 16x16 patches
            target_embed_dim=self.vit.embed_dim,  # Use ViT embedding dimension for prompts
            prompt_layers=self.prompt_layers,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            device=device
        )

        for n, p in self.vit.named_parameters():
            if n != 'head.weight' and n != 'head.bias':
                p.requires_grad = False  # 冻结除了分类头之外的所有参数
            else:
                p.requires_grad = True   # 只有分类头可训练


    def train(self, mode=True):
        super().train(False)
        self.prompter.train(False)
        self.vit.train(mode)
        return self

    def forward(self, x: torch.Tensor, cur_classes: int, frozen_past_classes=0, query_x=None, return_features=False, return_query=False):
        """
        Forward pass of the second stage model.
        """
        enable_renorm = query_x is None
        query_x = x if query_x is None else query_x
        clip_query = self.prompter.get_query(query_x, disable_renorm=not enable_renorm)
        # Ensure clip_query is in float32
        clip_query = clip_query.float()

        features = self.vit.forward_features(x, first_stage_query=clip_query, prompter=self.prompter, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes)
        
        if return_features:
            return features

        # out = self.vit.forward(x, first_stage_query=clip_query, prompter=self.prompter, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes)
        out = self.vit.forward_head(features)
        
        if return_query:
            return out, clip_query
        return out
    
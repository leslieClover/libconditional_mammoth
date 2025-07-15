"""
Second stage prompter for STAR-Prompt.
Handles CLIP-based prompting for the Vision Transformer.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import List
from kornia.augmentation import Normalize

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")


class Prompter(torch.nn.Module):
    """Second stage prompter for STAR-Prompt"""

    def __init__(self, args, num_classes: int, target_embed_len: int, target_embed_dim: int,
                 prompt_layers: List[int], clip_model: clip.model.CLIP = None,
                 clip_preprocess=None, device='cpu'):
        super().__init__()
        assert args.prompt_mode in ['residual', 'concat'], 'This prompter supports only residual or concat modes'

        self.args = args
        self.prompt_layers = prompt_layers
        self.target_embed_len = target_embed_len
        self.target_embed_dim = target_embed_dim
        self.device = device
        self.num_classes = num_classes
        self.prompt_mode = args.prompt_mode

        if clip_model is not None:
            assert clip_preprocess is not None, 'Preprocess must be provided if the model is provided'

        logging.info("Loading CLIP visual encoder and the pre-computed text features...")
        clip_backbone = 'ViT-L/14' if not hasattr(args, 'clip_backbone') else args.clip_backbone

        if clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
            self.clip_model = self.clip_model.float()
        else:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess

        # Store CLIP embedding dimension
        self.clip_embed_dim = self.clip_model.visual.output_dim

        # Initialize keys (will be set from first stage)
        self.keys = torch.zeros(num_classes, self.clip_embed_dim, device=device)

        # Setup CLIP preprocessing
        self.clip_normalization = Normalize(
            self.clip_preprocess.transforms[-1].mean,
            self.clip_preprocess.transforms[-1].std
        ).to(self.device)

        # Freeze CLIP model
        for p in self.clip_model.parameters():
            p.requires_grad = False



        # Initialize prompt parameters for each layer
        logging.info(f"Initializing prompts with target_embed_dim: {self.target_embed_dim}, clip_embed_dim: {self.clip_embed_dim}")
        
        for l in self.prompt_layers:
            if args.prompt_mode == 'residual':
                # Residual prompting - create prompts with ViT dimension
                tmp = self.get_parameter((self.num_classes, self.target_embed_dim))
                setattr(self, f'p_{l}', tmp)
                if l == 0:  # Only log for first layer
                    logging.info(f"Layer {l}: Created residual prompt with shape {tmp.shape}")
            else:
                # Prefix tuning prompting
                prompt_param = self.get_parameter((
                    self.num_classes,
                    2 * self.args.prefix_tuning_prompt_len,
                    self.target_embed_dim
                ))
                setattr(self, f'p_concat_{l}', prompt_param)
                if l == 0:  # Only log for first layer
                    logging.info(f"Layer {l}: Created concat prompt with shape {prompt_param.shape}")

            # Attention weights for each class - use CLIP embedding dimension
            attn_param = self.get_parameter((self.num_classes, self.clip_embed_dim))
            setattr(self, f'a_{l}', attn_param)
            if l == 0:  # Only log for first layer
                logging.info(f"Layer {l}: Created attention weights with shape {attn_param.shape}")

    def set_keys(self, keys: torch.Tensor, start_class: int, end_class: int):
        """Set the keys for the classes in the range [start_class, end_class)"""
        assert end_class - start_class == keys.shape[0], 'Number of classes in the keys tensor does not match the range'
        self.keys[start_class:end_class] = keys

    def get_parameter(self, shape, type_init: str = 'orto') -> torch.nn.Parameter:
        """Create and initialize a parameter tensor"""
        param = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=self.device))
        if type_init == 'orto':
            torch.nn.init.orthogonal_(param)
        elif type_init == 'gaussian':
            torch.nn.init.normal_(param, mean=0.0, std=0.1)
        return param

    @torch.no_grad()
    def get_query(self, x, disable_renorm=True):
        """Compute the CLIP features for the input image"""
        if not disable_renorm:
            # Apply CLIP normalization if needed
            x = self.clip_normalization(x)
        clip_out = self.clip_model.encode_image(x)
        return clip_out

    def compute_maps(self, clip_query: torch.Tensor, modulation_coeffs: torch.Tensor,
                     keys: torch.Tensor) -> torch.Tensor:
        """Compute the CLIP output given the clip_query and the keys"""
        filter_values = torch.softmax(modulation_coeffs, dim=-1)

        clip_query = clip_query.unsqueeze(1).expand(clip_query.shape[0], modulation_coeffs.shape[0],
                                                    clip_query.shape[-1])
        clip_out_a = clip_query * filter_values[None, :, :]
        clip_out_a_norm = torch.nn.functional.normalize(clip_out_a, dim=-1)

        clip_query = torch.einsum('bcd,cd->bc', clip_out_a_norm, keys) * 5
        return clip_query

    def get_masked_clip_out(self, sim_act_map):
        """Only keep the output of the CLIP model for the most similar class"""
        with torch.no_grad():
            mask = torch.ones_like(sim_act_map, dtype=torch.bool)
            mask.scatter_(1, sim_act_map.argmax(dim=1, keepdim=True), False)
            sim_act_map[mask] = 0.0
        return sim_act_map

    def compute_super_prompts(self, class_prompts: torch.Tensor, masked_clip_out: torch.Tensor, start_idx: int,
                              end_idx: int) -> torch.Tensor:
        """Compute the actual super-prompt by merging individual prompts"""
        masked_clip_out = masked_clip_out[:, start_idx:end_idx]
        class_prompts = class_prompts[start_idx:end_idx]

        if not self.args.enable_confidence_modulation:
            masked_clip_out = (masked_clip_out != 0).float()

        if self.args.prompt_mode == 'residual':
            sp = torch.einsum('bc,cd->bd', masked_clip_out, class_prompts)
        else:
            sp = torch.einsum('bc,cmd->bmd', masked_clip_out, class_prompts)
        return sp

    def get_prompts(self, layer_idx: int, clip_query: torch.Tensor, cur_classes: int, frozen_past_classes=0):
        """Compute the prompts for the layer_idx-th layer"""
        if layer_idx in self.prompt_layers:
            a: torch.Tensor = getattr(self, f'a_{layer_idx}')

            if self.prompt_mode == 'residual':
                pv: torch.Tensor = getattr(self, f'p_{layer_idx}')
                # Debug: log shapes only for first layer and first call
                if layer_idx == 0 and hasattr(self, '_debug_logged') is False:
                    logging.info(f"Debug - Layer {layer_idx}: clip_query shape: {clip_query.shape}")
                    logging.info(f"Debug - Layer {layer_idx}: attention weights shape: {a.shape}")
                    logging.info(f"Debug - Layer {layer_idx}: prompt params shape: {pv.shape}")
                    self._debug_logged = True
            else:
                p_concat: torch.Tensor = getattr(self, f'p_concat_{layer_idx}')
                p_concat_k, p_concat_v = torch.split(p_concat, self.args.prefix_tuning_prompt_len, dim=1)

            if frozen_past_classes > 0:
                with torch.no_grad():
                    clip_out_prev = self.compute_maps(clip_query, a[:frozen_past_classes].detach(),
                                                      self.keys[:frozen_past_classes].detach())
                clip_out_curr = self.compute_maps(clip_query, a[frozen_past_classes:cur_classes],
                                                  self.keys[frozen_past_classes:cur_classes])
                clip_out = torch.cat((clip_out_prev.detach(), clip_out_curr), dim=1)
                clip_out = self.get_masked_clip_out(clip_out)

                with torch.no_grad():
                    if self.prompt_mode == 'residual':
                        sp_past = self.compute_super_prompts(pv, clip_out, 0, frozen_past_classes)
                    else:
                        sp_concat_k_past = self.compute_super_prompts(p_concat_k, clip_out, 0,
                                                                      frozen_past_classes).squeeze(2)
                        sp_concat_v_past = self.compute_super_prompts(p_concat_v, clip_out, 0,
                                                                      frozen_past_classes).squeeze(2)

                if self.prompt_mode == 'residual':
                    sp_curr = self.compute_super_prompts(pv, clip_out, frozen_past_classes, cur_classes)
                    super_prompt = sp_past.detach() + sp_curr
                else:
                    sp_concat_k_curr = self.compute_super_prompts(p_concat_k, clip_out, frozen_past_classes,
                                                                  cur_classes).squeeze(2)
                    sp_concat_v_curr = self.compute_super_prompts(p_concat_v, clip_out, frozen_past_classes,
                                                                  cur_classes).squeeze(2)
                    super_prompt = (
                    sp_concat_k_past.detach() + sp_concat_k_curr, sp_concat_v_past.detach() + sp_concat_v_curr)
            else:
                clip_out = self.compute_maps(clip_query, a[:cur_classes], self.keys[:cur_classes])
                clip_out = self.get_masked_clip_out(clip_out)

                if self.prompt_mode == 'residual':
                    super_prompt = self.compute_super_prompts(pv, clip_out, 0, cur_classes)
                else:
                    sp_concat_k = self.compute_super_prompts(p_concat_k, clip_out, 0, cur_classes).squeeze(2)
                    sp_concat_v = self.compute_super_prompts(p_concat_v, clip_out, 0, cur_classes).squeeze(2)
                    super_prompt = (sp_concat_k, sp_concat_v)

            return super_prompt
        else:
            return None

    def compute_ortho_loss(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:
        """Compute the orthogonality loss for the prompts"""
        if frozen_past_classes == 0:
            return torch.tensor(0.0, device=self.device)

        ortho_loss_list = []
        weight_loss_list = []

        def _compute_loss(p: torch.Tensor, frozen_past_classes: int, cur_classes: int) -> torch.Tensor:
            past_pv = p[:frozen_past_classes].detach()
            cur_pv = p[frozen_past_classes:cur_classes]

            # Flatten if necessary for proper matrix multiplication
            if cur_pv.dim() > 2:
                cur_pv_flat = cur_pv.view(cur_pv.shape[0], -1)
                past_pv_flat = past_pv.view(past_pv.shape[0], -1)
            else:
                cur_pv_flat = cur_pv
                past_pv_flat = past_pv

            eye_intra = torch.eye(cur_classes - frozen_past_classes, device=cur_pv.device).bool()
            intra_ortho_loss = (torch.matmul(cur_pv_flat, cur_pv_flat.T)[eye_intra] - 1).pow(2).mean()
            inter_ortho_loss = (torch.matmul(cur_pv_flat, past_pv_flat.T)).pow(2).mean()
            return intra_ortho_loss + inter_ortho_loss

        for layer_idx in self.prompt_layers:
            if self.prompt_mode == 'residual':
                p = getattr(self, f'p_{layer_idx}')
                current_loss = _compute_loss(p, frozen_past_classes, cur_classes)
            else:
                p_concat = getattr(self, f'p_concat_{layer_idx}')
                p_concat_k, p_concat_v = torch.split(p_concat, self.args.prefix_tuning_prompt_len, dim=1)

                p_concat_k = p_concat_k.view(p_concat_k.shape[0], -1)
                p_concat_v = p_concat_v.view(p_concat_v.shape[0], -1)

                current_loss_k = _compute_loss(p_concat_k, frozen_past_classes, cur_classes)
                current_loss_v = _compute_loss(p_concat_v, frozen_past_classes, cur_classes)
                current_loss = current_loss_k + current_loss_v

            current_weight = 1.
            if layer_idx < self.args.ortho_split_val:
                current_weight = 0.

            current_loss = current_weight * current_loss
            weight_loss_list.append(current_weight)
            ortho_loss_list.append(current_loss)

        total_ortho_loss = sum(ortho_loss_list) / sum(weight_loss_list) if sum(weight_loss_list) > 0 else torch.tensor(
            0.0, device=self.device)
        return total_ortho_loss
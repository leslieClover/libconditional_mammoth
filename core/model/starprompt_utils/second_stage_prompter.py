"""
Second stage prompter for STAR-Prompt.
Handles CLIP-based prompting for the Vision Transformer.
"""
from argparse import Namespace
import os
import logging
import torch
import torch.nn as nn
from typing import List
from kornia.augmentation import Normalize
from .templates import templates
import json
# from kornia.enhance import Denormalize 

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")


class DeNormalize(object):
    def __init__(self, mean, std):
        """
        完全复制 Mammoth 的 DeNormalize 实现
        """
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean)
        elif isinstance(mean, torch.Tensor):
            mean = mean.clone()
        
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std)
        elif isinstance(std, torch.Tensor):
            std = std.clone()

        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        与 Mammoth 完全一致的反归一化实现
        """
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        # 确保设备一致性
        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)

        # 关键：使用正确的广播形状
        mean = self.mean.view(1, -1, 1, 1) if self.mean.dim() == 1 else self.mean[:, None, None]
        std = self.std.view(1, -1, 1, 1) if self.std.dim() == 1 else self.std[:, None, None]
        
        return (tensor * std) + mean

    def to(self, device):
        """设备转移方法"""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


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

        # if clip_model is None:
        #     self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
        #     self.clip_model = self.clip_model.float()
        # else:
        #     self.clip_model = clip_model
        #     self.clip_preprocess = clip_preprocess

        # # Store CLIP embedding dimension
        # self.clip_embed_dim = self.clip_model.visual.output_dim

        # # Initialize keys (will be set from first stage)
        # self.keys = torch.zeros(num_classes, self.clip_embed_dim, device=device)


        # ！！！ 改了这一段， ⭐ 核心部分：复现mammoth的keys加载逻辑
        self.keys_ckpt_path = None
        self.old_args = None
        
        if hasattr(args, 'keys_ckpt_path') and args.keys_ckpt_path is not None:
            # 第一阶段有checkpoint的情况
            self.keys_ckpt_path = self._resolve_keys_path(args)
            
            if self.keys_ckpt_path and os.path.exists(self.keys_ckpt_path):
                logging.info(f"Loading keys from checkpoint: {self.keys_ckpt_path}")
                self.keys, first_stage_args = self.load_keys()
                if first_stage_args is not None:
                    logging.info(f"Keys loaded. Loading CLIP version: {first_stage_args.clip_backbone}")
                    clip_backbone = first_stage_args.clip_backbone
                    
                if clip_model is None:
                    self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
                    self.clip_model = self.clip_model.float()
                else:
                    self.clip_model = clip_model
                    self.clip_preprocess = clip_preprocess
            else:
                # checkpoint不存在，使用默认模板
                logging.warning(f"Keys checkpoint not found: {self.keys_ckpt_path}, using default templates")
                self._initialize_with_templates(clip_model, clip_preprocess, clip_backbone)
        else:
            self.keys_ckpt_path = None
            # 没有指定checkpoint，使用默认模板
            logging.info("No keys checkpoint specified, using default CLIP templates")
            self._initialize_with_templates(clip_model, clip_preprocess, clip_backbone)


        # Setup CLIP preprocessing
        self.clip_normalization = Normalize(
            self.clip_preprocess.transforms[-1].mean,
            self.clip_preprocess.transforms[-1].std
        ).to(self.device)


        if hasattr(args, 'train_trfms') and args.train_trfms:
            # 从训练变换中找到 Normalize 配置
            normalize_config = None
            for transform in args.train_trfms:
                if 'Normalize' in transform:
                    normalize_config = transform['Normalize']
                    break
            
            if normalize_config:
                mean_values = normalize_config['mean']
                std_values = normalize_config['std']
            else:
                # 如果没找到，使用 CIFAR100 默认值
                mean_values = [0.5071, 0.4867, 0.4408]
                std_values = [0.2675, 0.2565, 0.2761]
        
        # 方案2: 如果 args 中直接有数据集特定的配置
        elif hasattr(args, 'dataset_mean') and hasattr(args, 'dataset_std'):
            mean_values = args.dataset_mean
            std_values = args.dataset_std
        
        # 方案3: 根据数据集名称选择对应的参数
        elif hasattr(args, 'dataset'):
            dataset_name = args.dataset.lower()
            if 'cifar100' in dataset_name:
                mean_values = [0.5071, 0.4867, 0.4408]
                std_values = [0.2675, 0.2565, 0.2761]
            elif 'imagenet' in dataset_name:
                mean_values = [0.485, 0.456, 0.406]
                std_values = [0.229, 0.224, 0.225]
            else:
                # 默认使用 YAML 中配置的 ImageNet 参数
                mean_values = [0.485, 0.456, 0.406]
                std_values = [0.229, 0.224, 0.225]
        else:
            # 最终默认值：使用 YAML 中的 ImageNet 参数
            mean_values = [0.485, 0.456, 0.406]
            std_values = [0.229, 0.224, 0.225]

        self.denorm_transform = DeNormalize(mean=mean_values, std=std_values).to(device)

        # Freeze CLIP model
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # 获取CLIP embedding dimension
        self.clip_embed_dim = self.clip_model.visual.output_dim
        logging.info(f"CLIP embedding dimension: {self.clip_embed_dim}")

        for l in self.prompt_layers:
            if args.prompt_mode == 'residual':
                # NOTE: this initialization follows that of CODA-Prompt.
                # We originally initialize a prompt for key, query, and value of the MHA layer.
                tmp = self.get_parameter((self.num_classes, 3, self.target_embed_dim))
                # We only use value at the end, so we keep only a single tensor.
                tmp.data = tmp.data[:, 0]
                # HOWEVER: Since the orthogonal_ of pytorch flattens the tensor, the value prompt is not orthogonal anymore.
                # orthogonal_ made (C, 3, D) -> (C, 3*D) -> orthogonal -> (C, 3, D), thus each 3*D is orthogonal, but not each D.
                # This is intended and makes the orthogonalization loss being optimized at the beginning.
                setattr(self, f'p_{l}', tmp)
                if l == 0:  # Only log for first layer
                    logging.info(f"Layer {l}: Created CODA-style residual prompt with shape {tmp.shape} (originally 3D)")
            else:
                # Prefix tuning prompting - keep original implementation
                prompt_param = self.get_parameter((
                    self.num_classes,
                    2 * self.args.prefix_tuning_prompt_len,
                    self.target_embed_dim
                ))
                setattr(self, f'p_concat_{l}', prompt_param)
                if l == 0:  # Only log for first layer
                    logging.info(f"Layer {l}: Created concat prompt with shape {prompt_param.shape}")

            # Attention weights for each class - 保持原有实现
            attn_param = self.get_parameter((self.num_classes, self.clip_embed_dim))
            setattr(self, f'a_{l}', attn_param)
            if l == 0:  # Only log for first layer
                logging.info(f"Layer {l}: Created attention weights with shape {attn_param.shape}")


    def _resolve_keys_path(self, args):
        """解析keys checkpoint路径"""
        keys_ckpt_path = args.keys_ckpt_path
        
        if keys_ckpt_path.endswith('.json'):
            # JSON格式：{dataset: {seed: job_id}}
            try:
                with open(keys_ckpt_path, 'r') as f:
                    key_mapping = json.load(f)
                
                dataset_name = getattr(args, 'dataset', 'unknown')
                seed = str(getattr(args, 'seed', 1993))
                
                if dataset_name in key_mapping and seed in key_mapping[dataset_name]:
                    key_jobnum = key_mapping[dataset_name][seed]
                    # 构造文件路径（简化版本，不依赖dataset.N_TASKS）
                    resolved_path = f"coop_keys/coop_keys_{key_jobnum}.pt"
                else:
                    logging.warning(f"Key mapping not found for dataset: {dataset_name}, seed: {seed}")
                    return None
                    
            except Exception as e:
                logging.error(f"Error reading JSON keys file: {e}")
                return None
                
        elif keys_ckpt_path.endswith('.pt'):
            # 直接的.pt文件
            resolved_path = keys_ckpt_path
        else:
            # job_id字符串，构造文件路径
            resolved_path = f"coop_keys/coop_keys_{keys_ckpt_path}.pt"
            
        return resolved_path
    
    def _initialize_with_templates(self, clip_model, clip_preprocess, clip_backbone):
        """使用默认模板初始化"""
        if clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load(clip_backbone, self.device)
            self.clip_model = self.clip_model.float()
        else:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess
            
        # 生成默认的类名
        dataset_classes = [f"class_{i}" for i in range(self.num_classes)]
        
        # 生成默认的prompt模板, 这里的内容在 mammoth 中
        default_templates = templates['imagenet']
        
        self.keys = self.load_default_prompt_templates(default_templates, dataset_classes)

    
    @torch.no_grad()
    def load_default_prompt_templates(self, templates: List[str], dataset_classes: List[str]) -> torch.Tensor:
        """从mammoth复制的函数"""
        if hasattr(self.args, 'statc_keys_use_templates') and self.args.statc_keys_use_templates:
            all_features = []
            for t in templates:
                text_inputs = torch.cat([clip.tokenize(t.format(c)) for c in dataset_classes]).to(self.device)
                text_features = self.clip_model.encode_text(text_inputs)
                all_features.append(text_features)
            text_features = torch.stack(all_features).mean(dim=0)
        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset_classes]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    @torch.no_grad()
    def load_keys(self):
        """从mammoth复制的函数"""
        if not self.keys_ckpt_path or not os.path.exists(self.keys_ckpt_path):
            logging.warning(f"Keys checkpoint not found: {self.keys_ckpt_path}")
            return None, None
            
        logging.info(f'Loading keys from {self.keys_ckpt_path}')
        try:
            st = torch.load(self.keys_ckpt_path, weights_only=True)
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            return None, None
        
        if isinstance(st, dict):
            keys = st['keys'].to(self.device)
            try:
                old_args = Namespace(**st['args'])
                # 验证兼容性
                assert self.num_classes == keys.shape[0], f"Class number mismatch: {self.num_classes} vs {keys.shape[0]}"
                # 可以添加更多验证...
            except Exception as e:
                logging.warning(f"Args validation failed: {e}")
                old_args = None
        else:
            keys = st.to(self.device)
            old_args = None
            assert self.num_classes == keys.shape[0], f"Class number mismatch: {self.num_classes} vs {keys.shape[0]}"
            
        logging.info(f'Keys loaded successfully with shape: {keys.shape}')
        return keys.float(), old_args


    def set_keys(self, keys: torch.Tensor, start_class: int, end_class: int):
        """Set the keys for the classes in the range [start_class, end_class)"""
        assert end_class - start_class == keys.shape[0], 'Number of classes in the keys tensor does not match the range'
        self.keys[start_class:end_class] = keys

    def update_keys_from_first_stage(self, first_stage_keys: torch.Tensor):
        """从第一阶段更新所有keys（LibContinual特有的动态更新方法）"""
        self.keys = first_stage_keys.clone().to(self.device)
        logging.info(f"Updated all keys from first stage, shape: {self.keys.shape}")

    

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
            x = self.denorm_transform(x)
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
                # if layer_idx == 0 and hasattr(self, '_debug_logged') is False:
                #     logging.info(f"Debug - Layer {layer_idx}: clip_query shape: {clip_query.shape}")
                #     logging.info(f"Debug - Layer {layer_idx}: attention weights shape: {a.shape}")
                #     logging.info(f"Debug - Layer {layer_idx}: prompt params shape: {pv.shape}")
                #     self._debug_logged = True
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

            # Flatten if necessary for proper matrix multiplication, 在这里做的修复,避免了潜在的 device 不匹配错误
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
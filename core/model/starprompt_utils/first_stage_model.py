"""
First stage model for STAR-Prompt.
Implements CLIP-based prompting for text-image alignment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from typing import List
from tqdm.auto import tqdm

from .generative_replay import MixtureOfGaussiansModel
from core.utils.conf import create_seeded_dataloader

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")

try:
    import wandb
except ImportError:
    wandb = None


class TextEncoder(torch.nn.Module):
    """Text encoder for CLIP model"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Prompter(torch.nn.Module):
    """CLIP-based prompter for first stage"""

    def __init__(self, args, num_classes: int, device='cpu'):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device = device

        self.clip_model, self.clip_preprocess = clip.load(args.clip_backbone, self.device)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Create class names (simplified for framework compatibility)
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.setup_text_prompting()
        self.clip_logit_scale = self.clip_model.logit_scale

        embed_dim = self.clip_model.visual.output_dim
        self.distributions = torch.nn.ModuleList([
            MixtureOfGaussiansModel(
                embed_dim,
                n_components=self.args.gr_mog_n_components,
                n_iters=self.args.gr_mog_n_iters_first_stage
            ) for _ in range(self.num_classes)
        ]).to(self.device)

    def compute_ortho_loss(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:
        """Compute orthogonality loss for prompt parameters"""
        cur_coop_p = self.prompt_parameters[frozen_past_classes:cur_classes]
        ortho_loss_coop = torch.tensor(0.0, device=self.device)

        if frozen_past_classes > 0:
            past_coop_p = self.prompt_parameters[:frozen_past_classes].detach()
            ortho_loss_coop = (torch.matmul(cur_coop_p.permute(1, 0, 2), past_coop_p.permute(1, 2, 0)) ** 2).mean()

        return ortho_loss_coop

    @torch.no_grad()
    def create_features_dataset(self, current_task: int):
        """Create features dataset for generative replay"""
        labels, features = [], []

        for _ti in range(current_task + 1):
            prev_t_size, cur_t_size = self.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):
                current_samples = self.distributions[class_idx](self.args.num_samples_gr)
                features.append(current_samples)
                labels.append(torch.ones((self.args.num_samples_gr)) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()
        return create_seeded_dataloader(
            self.args,
            TensorDataset(features, labels),
            num_workers=0,
            batch_size=self.args.batch_size_gr,
            shuffle=True,
            non_verbose=True
        )

    def get_offsets(self, task_id):
        """Get class offsets for a given task"""
        # 这里的函数有没有？
        if task_id == 0:
            return 0, self.args.init_cls_num if hasattr(self.args, 'init_cls_num') else 10
        else:
            init_cls = self.args.init_cls_num if hasattr(self.args, 'init_cls_num') else 10
            inc_cls = self.args.inc_cls_num if hasattr(self.args, 'inc_cls_num') else 10
            return init_cls + (task_id - 1) * inc_cls, init_cls + task_id * inc_cls

    def train_alignment_epoch(self, optim: torch.optim.Optimizer, current_task: int, epoch: int = 0):
        """Train alignment epoch for generative replay"""
        offset_1, offset_2 = self.get_offsets(current_task)
        dl = self.create_features_dataset(current_task)

        with tqdm(enumerate(dl), total=len(dl),
                  desc=f'GR first stage epoch {epoch + 1}/{self.args.num_epochs_gr_first_stage}', leave=False) as pbar:
            for i, (image_features, labels) in pbar:
                if self.args.debug_mode and i > 3:
                    break
                optim.zero_grad()

                image_features = image_features.to(self.device, dtype=self.clip_model.dtype)
                labels = labels.to(self.device)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

                text_features = self.compute_keys(0, offset_2)
                text_features = torch.cat((text_features[:offset_1].detach(), text_features[offset_1:offset_2]), dim=0)
                text_features = torch.nn.functional.normalize(text_features, dim=-1)

                clip_logits = torch.einsum('bd,cd->bc', image_features, text_features)
                clip_logits = clip_logits * self.clip_logit_scale.exp()
                loss = F.cross_entropy(clip_logits, labels)

                assert not math.isnan(loss.item())
                loss.backward()
                optim.step()

                pbar.set_postfix({'loss': loss.item()}, refresh=False)

                if not self.args.nowand and wandb is not None:
                    assert wandb is not None, "wandb is not installed."
                    wandb.log({'ca_loss_first_stage': loss.item(), 'ca_lr_first_stage': optim.param_groups[0]['lr']})

    def align(self, current_task: int):
        """Align prompts using generative replay"""
        optim = torch.optim.SGD(
            lr=self.args.learning_rate_gr_first_stage,
            params=[self.prompt_parameters],
            momentum=0.0,
            weight_decay=0.0
        )

        for e in range(self.args.num_epochs_gr_first_stage):
            self.train_alignment_epoch(optim, current_task=current_task, epoch=e)

    @torch.no_grad()
    def update_statistics(self, dataset, current_task: int):
        """Update statistics for generative replay"""
        offset_1, offset_2 = self.get_offsets(current_task)
        features_dict = {i: [] for i in range(offset_1, offset_2)}

        was_training = self.training
        self.eval()

        with tqdm(total=self.args.num_monte_carlo_gr_first_stage * len(dataset.train_loader),
                  desc='Updating statistics for first stage Generative Replay') as pbar:
            for _ in range(self.args.num_monte_carlo_gr_first_stage):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3 and min(
                            [len(v) for k, v in features_dict.items()]) > self.args.gr_mog_n_components:
                        break

                    inputs, labels = data['image'].to(self.device), data['label'].to(self.device, dtype=torch.long)

                    if len(inputs.shape) == 5:
                        inputs = inputs[:, 1]
                    clip_query = self.get_query(inputs)

                    for class_idx in labels.unique():
                        features_dict[int(class_idx)].append(clip_query[labels == class_idx])

                    pbar.update(1)

        for class_idx in range(offset_1, offset_2):
            features_class_idx = torch.cat(features_dict[class_idx], dim=0)
            self.distributions[class_idx].fit(features_class_idx.to(self.device))

        if was_training:
            self.train()

    def compute_keys(self, start: int, end: int):
        """Compute the text-encoder features for classes from start to end"""
        ctx = self.prompt_parameters[start:end]
        prefix = self.token_prefix[start:end]
        suffix = self.token_suffix[start:end]
        prompts = torch.cat((prefix, ctx, suffix), dim=1)
        tokenized_prompts = self.tokenized_prompts[start:end]
        keys = self.text_encoder(prompts.to(self.clip_model.dtype), tokenized_prompts)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        return keys

    def get_keys(self, cur_classes: int, frozen_past_classes=0) -> torch.Tensor:
        """Get text-encoder features for classes from 0 to cur_classes"""
        if frozen_past_classes > 0:
            with torch.no_grad():
                past_keys = self.compute_keys(0, frozen_past_classes)
            cur_keys = self.compute_keys(frozen_past_classes, cur_classes)
            keys = torch.cat((past_keys.detach(), cur_keys), dim=0)
        else:
            keys = self.compute_keys(0, cur_classes)
        return keys

    def setup_text_prompting(self):
        """Initialize prompts for each class"""
        self.text_encoder = TextEncoder(self.clip_model)

        text_prompts = ["X " + name + "." for name in self.class_names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts], dim=0).to(self.device)
        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 2:, :])  # CLS, EOS

        prompt_parameters = torch.empty(
            self.num_classes, 1,
            self.clip_model.token_embedding.weight.shape[1],
            device=self.device,
            dtype=torch.float32
        )
        torch.nn.init.normal_(prompt_parameters, std=0.02)
        self.prompt_parameters = torch.nn.Parameter(prompt_parameters)

    @torch.no_grad()
    def get_query(self, x):
        """Get CLIP visual features for input images"""
        clip_out = self.clip_model.encode_image(x)
        assert not torch.isnan(clip_out).any()
        return clip_out

    def get_clip_logits(self, clip_out, keys):
        """Get CLIP logits from visual features and text keys"""
        image_features = torch.nn.functional.normalize(clip_out, dim=-1)
        clip_logits = torch.einsum('bd,cd->bc', image_features, keys)
        clip_logits = clip_logits * self.clip_logit_scale.exp()
        return clip_logits


class FirstStageModel(torch.nn.Module):
    """First stage model for STAR-Prompt"""

    def __init__(self, args, num_classes: int, device='cpu'):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device = device

        self.prompter = Prompter(args, num_classes=num_classes, device=device)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.prompter.to(device, *args, **kwargs)
        self.device = device
        return self

    def train(self, mode=True):
        super().train(False)
        self.prompter.train(False)
        return self

    def forward(self, x: torch.Tensor, cur_classes: int, return_query=False, frozen_past_classes=0) -> torch.Tensor:
        """
        Forward pass of the first stage model.
        Returns CLIP logits or query features.
        """
        clip_out = self.prompter.get_query(x)
        if return_query:
            return clip_out

        keys = self.prompter.get_keys(frozen_past_classes=frozen_past_classes, cur_classes=cur_classes)
        clip_logits = self.prompter.get_clip_logits(clip_out, keys)

        return clip_logits

    def train_first_stage_on_task(self, dataset, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
        """Train the first stage on current task"""
        # 这个代码也在原来的mammoth-master中 没有。
        logging.info(f"Starting training of first stage on task: {current_task}")

        # Setup training
        was_training = self.training
        self.train()

        # Get trainable parameters
        first_stage_params = [v for k, v in self.named_parameters() if 'prompt_parameters' in k]

        if self.args.first_stage_optim == 'sgd':
            opt = torch.optim.SGD(
                first_stage_params,
                lr=self.args.first_stage_lr,
                momentum=self.args.first_stage_momentum,
                weight_decay=self.args.first_stage_weight_decay
            )
        else:
            opt = torch.optim.Adam(
                first_stage_params,
                lr=self.args.first_stage_lr,
                weight_decay=self.args.first_stage_weight_decay
            )

        # Training loop
        with tqdm(total=self.args.first_stage_epochs * len(dataset.train_loader), desc='First stage training') as pbar:
            for epoch in range(self.args.first_stage_epochs):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3:
                        break

                    inputs, labels = data['image'].to(self.device), data['label'].to(self.device, dtype=torch.long)
                    loss = torch.tensor(0.).to(self.device)

                    opt.zero_grad()

                    # Forward pass
                    clip_logits = self(inputs, frozen_past_classes=n_past_classes, cur_classes=n_seen_classes)

                    # Mask past classes
                    clip_logits[:, :n_past_classes] = -float('inf')
                    loss_clip = loss_fn(clip_logits[:, :n_seen_classes], labels)
                    loss += loss_clip

                    # Orthogonality loss
                    loss_ortho_coop = self.prompter.compute_ortho_loss(
                        frozen_past_classes=n_past_classes,
                        cur_classes=n_seen_classes
                    )
                    loss += self.args.lambda_ortho_first_stage * loss_ortho_coop

                    # Backward pass
                    if i == 0:
                        opt.zero_grad()
                    (loss / self.args.virtual_bs_n).backward()
                    if (i > 0 or self.args.virtual_bs_n == 1) and i % self.args.virtual_bs_n == 0:
                        opt.step()
                        opt.zero_grad()

                    # Logging
                    if not self.args.nowand and wandb is not None:
                        wandb.log({
                            'first_stage_loss': loss.item(),
                            'first_stage_lr': opt.param_groups[0]['lr'],
                            'first_stage_epoch': epoch,
                            'first_stage_loss_clip': loss_clip.item(),
                            'first_stage_loss_ortho': loss_ortho_coop.item(),
                            'first_stage_iteration': i
                        })

                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()}, refresh=False)

        # Cleanup
        opt.zero_grad(set_to_none=True)
        del opt
        torch.cuda.empty_cache()

        # Generative replay
        if self.args.enable_gr:
            self.prompter.update_statistics(dataset, current_task)
            self.prompter.align(current_task)

        self.train(was_training)


import logging
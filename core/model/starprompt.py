# -*- coding: utf-8 -*-
"""
STAR-Prompt: Spatial-Temporal Adaptive Reasoning for Large-scale Continual Learning
Adapted for LibContinual framework from the original implementation.

Original paper: https://arxiv.org/abs/2306.09200
Original code: https://github.com/aimagelab/mammoth/tree/master/models/star_prompt_utils
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Union, Tuple

from .finetune import Finetune
from .starprompt_utils.first_stage_model import FirstStageModel
from .starprompt_utils.second_stage_model import SecondStageModel
from .starprompt_utils.generative_replay import Gaussian, MixtureOfGaussiansModel
from core.utils.utils import count_parameters
from core.utils.conf import create_seeded_dataloader
from core.scheduler import CosineSchedule


try:
    import wandb
except ImportError:
    wandb = None

try:
    import clip
    from clip.model import convert_weights
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")


class STARPromptModel(nn.Module):
    """
    STAR-Prompt model consisting of two stages:
    1. First stage: CLIP-based prompting for text-image alignment
    2. Second stage: Vision Transformer with prompting for final classification
    """

    def __init__(self, args, backbone: nn.Module, num_classes: int, device='cpu'):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device = device

        # Initialize first stage
        logging.info("Loading first stage...")
        self.first_stage = FirstStageModel(args=args, num_classes=num_classes, device=device)

        # Remove track running stats from CLIP
        for m in self.first_stage.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False

        # Initialize second stage
        logging.info("Loading second stage...")
        self.second_stage = SecondStageModel(
            args=args,
            num_classes=num_classes,
            backbone=backbone,
            clip_model=self.first_stage.prompter.clip_model,
            clip_preprocess=self.first_stage.prompter.clip_preprocess,
            device=device
        )

        # Initialize generative replay distributions
        embed_dim = self.second_stage.vit.embed_dim
        self.second_stage_distributions = torch.nn.ModuleList([
            self._get_dist(embed_dim) for _ in range(self.num_classes)
        ]).to(self.device)

        self.classifier_state_dict = None
        logging.info("STAR-Prompt model initialized successfully.")

    def _get_dist(self, embed_dim):
        """Get distribution for generative replay"""
        if self.args.gr_model == 'mog':
            return MixtureOfGaussiansModel(
                embed_dim,
                n_components=self.args.gr_mog_n_components,
                n_iters=self.args.gr_mog_n_iters_second_stage
            )
        else:
            return Gaussian(embed_dim)

    def forward(self, x: torch.Tensor, cur_classes: int, frozen_past_classes=0, return_query=False):
        """Complete forward pass of STAR-Prompt"""
        return self.second_stage(x, cur_classes=cur_classes, frozen_past_classes=frozen_past_classes,
                                 return_query=return_query)

    def train(self, mode: bool = True):
        self.first_stage.train(mode)
        self.second_stage.train(mode)
        return self

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.first_stage.to(device, *args, **kwargs)
        self.second_stage.to(device, *args, **kwargs)
        self.device = device
        return self

    @torch.no_grad()
    def update_keys(self, start_c: int, end_c: int):
        """Update keys for second stage from first stage"""
        logging.info('Updating keys for second stage...')
        first_stage_keys = self.first_stage.prompter.compute_keys(start_c, end_c)
        self.second_stage.prompter.set_keys(first_stage_keys, start_c, end_c)

    def norm(self, t):
        """Normalize tensor"""
        return torch.norm(t, p=2, dim=-1, keepdim=True) + 1e-7

    @torch.no_grad()
    def create_features_dataset(self, current_task: int):
        """Create features dataset for generative replay"""
        labels, features = [], []

        for _ti in range(current_task + 1):
            prev_t_size, cur_t_size = self.get_offsets(_ti)

            for class_idx in range(prev_t_size, cur_t_size):
                current_samples = self.second_stage_distributions[class_idx](self.args.num_samples_gr)
                features.append(current_samples)
                labels.append(torch.ones(self.args.num_samples_gr) * class_idx)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return create_seeded_dataloader(
            self.args,
            TensorDataset(features, labels),
            batch_size=self.args.batch_size_gr,
            shuffle=True,
            num_workers=0,
            non_verbose=True
        )

    def get_offsets(self, task_id):
        """Get class offsets for a given task"""
        if task_id == 0:
            return 0, getattr(self.args, 'init_cls_num', 10)
        else:
            init_cls = getattr(self.args, 'init_cls_num', 10)
            inc_cls = getattr(self.args, 'inc_cls_num', 10)
            return init_cls + (task_id - 1) * inc_cls, init_cls + task_id * inc_cls

    def train_alignment_epoch(self, classifier: torch.nn.Module, optim: torch.optim.Optimizer, n_seen_classes: int,
                              current_task: int, loss_fn):
        """Train alignment epoch for generative replay"""
        dl = self.create_features_dataset(current_task)

        with tqdm(enumerate(dl), total=len(dl), desc='GR epoch') as pbar:
            for i, (x, labels) in pbar:
                optim.zero_grad()
                x, labels = x.to(self.device, dtype=torch.float32), labels.to(self.device)

                logits = classifier(x)
                logits = logits[:, :n_seen_classes]

                norm = self.norm(logits)
                logits = logits / (0.1 * norm)

                loss = loss_fn(logits, labels)
                print(173)
                loss.backward()
                print(176)
                optim.step()

                if not self.args.nowand and wandb is not None:
                    wandb.log({'ca_loss_second_stage': loss.item(), 'ca_lr_second_stage': optim.param_groups[0]['lr']})
                pbar.set_postfix({'loss': loss.item()}, refresh=False)

    def align(self, current_task: int, n_seen_classes: int, loss_fn):
        """Align classifier using generative replay"""
        classifier = deepcopy(self.second_stage.vit.head)

        optim = torch.optim.SGD(
            lr=self.args.learning_rate_gr_second_stage,
            params=classifier.parameters(),
            momentum=0.0,
            weight_decay=0.0
        )

        num_epochs = self.args.num_epochs_gr_second_stage + (5 * current_task)

        for e in range(num_epochs):
            self.train_alignment_epoch(classifier, optim, n_seen_classes=n_seen_classes, current_task=current_task,
                                       loss_fn=loss_fn)

        self.second_stage.vit.head.weight.data.copy_(classifier.weight.data)
        self.second_stage.vit.head.bias.data.copy_(classifier.bias.data)

    @torch.no_grad()
    def update_statistics(self, dataset, n_past_classes: int, n_seen_classes: int):
        """Update statistics for generative replay"""
        features_dict = {i: [] for i in range(n_past_classes, n_seen_classes)}

        self.second_stage.eval()

        with tqdm(total=self.args.num_monte_carlo_gr_second_stage * len(dataset.train_loader),
                  desc='GR update statistics') as pbar:
            for _ in range(self.args.num_monte_carlo_gr_second_stage):
                for i, data in enumerate(dataset.train_loader):
                    if self.args.debug_mode and i > 3 and min(
                            [len(v) for k, v in features_dict.items()]) > self.args.gr_mog_n_components:
                        break

                    x, labels = data['image'], data['label']
                    x, labels = x.to(self.device), labels.to(self.device, dtype=torch.long)
                    features = self.second_stage(x, return_features=True, cur_classes=n_seen_classes,
                                                 frozen_past_classes=n_past_classes)
                    features = features[:, 0]

                    for class_idx in labels.unique():
                        features_dict[int(class_idx)].append(features[labels == class_idx])

                    pbar.update(1)

        for class_idx in range(n_past_classes, n_seen_classes):
            features_class_idx = torch.cat(features_dict[class_idx], dim=0)
            self.second_stage_distributions[class_idx].fit(features_class_idx.to(self.device))

    def backup(self, current_task: int, n_past_classes: int, n_seen_classes: int):
        """Backup classifier state"""
        logging.info(f"BACKUP: Task - {current_task} - classes from {n_past_classes} - to {n_seen_classes}")
        self.classifier_state_dict = deepcopy(self.second_stage.vit.head.state_dict())

    def recall_classifier_second_stage(self, current_task: int, n_past_classes: int, n_seen_classes: int):
        """Recall classifier state for second stage"""
        logging.info(f"RECALL: Task - {current_task} - classes from {n_past_classes} - to {n_seen_classes}")

        if current_task == 0 or not self.args.enable_gr:
            return

        assert self.classifier_state_dict

        self.second_stage.vit.head.weight.data.copy_(self.classifier_state_dict['weight'].data)
        self.second_stage.vit.head.bias.data.copy_(self.classifier_state_dict['bias'].data)

    # @torch.enable_grad()
    # def train_first_stage_on_task(self, dataset, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
    #     """Train the first stage on the current task"""
    #     return self.first_stage.train_first_stage_on_task(dataset, current_task, n_past_classes, n_seen_classes,
    #                                                       loss_fn)
    @torch.enable_grad()
    def train_first_stage_on_task(self, dataset, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
        """Train the first stage on the current task - 完全复现Mammoth逻辑"""
        
        logging.info(f"Starting training of first stage on task: {current_task}")
        
        # 步骤1: 保存原始transforms
        # print(259)
        train_loader = dataset if hasattr(dataset, '__iter__') else dataset.train_loader


        # print(263)
        # LibContinual数据格式处理 - 需要获取实际的dataset对象
        if hasattr(train_loader, 'dataset'):
            # print(266)
            actual_dataset = train_loader.dataset
            old_train_transform = actual_dataset.trfms if hasattr(actual_dataset, 'trfms') else None
        else:
            # 如果是简单的DataLoader
            # print(272)
            old_train_transform = None
            actual_dataset = None
        
        try:
            # 步骤2: 切换到CLIP预处理 (关键复现点)
            # print(278)
            if actual_dataset is not None:
                logging.info("Switching to CLIP preprocessing for first stage training")
                actual_dataset.trfms = self.first_stage.prompter.clip_preprocess
            
            # 步骤3: 设置CLIP模型为float16 (精确复现Mammoth)
            # print(283)

            convert_weights(self.first_stage.prompter.clip_model)
            self.first_stage.prompter.text_encoder.dtype = torch.float16
            # print(287)
            
            # 步骤4: 训练状态管理
            was_training = self.first_stage.training
            self.first_stage.train()
            
            # 步骤5: 获取可训练参数 (精确复现Mammoth逻辑)
            # print(295)
            first_stage_params = [v for k, v in self.first_stage.named_parameters() if 'prompt_parameters' in k]
            # print(298)
            if len(first_stage_params) == 0:
                logging.warning("No 'prompt_parameters' found, searching for alternative parameter names...")
                # 扩展搜索范围
                # print(301)
                alt_names = ['prompt', 'context', 'learnable', 'adapter', 'text_encoder']
                for name, param in self.first_stage.named_parameters():
                    if any(alt in name.lower() for alt in alt_names) and param.requires_grad:
                        first_stage_params.append(param)
                        logging.info(f"Found trainable parameter: {name}")
            # print(308)
            if len(first_stage_params) == 0:
                raise RuntimeError("Cannot find trainable parameters in first stage!")
            
            # print(310)
            
            # 步骤6: 创建优化器 (完全复现Mammoth)
            if self.args.first_stage_optim == 'sgd':
                # print(315)
                opt = torch.optim.SGD(first_stage_params, 
                                    lr=self.args.first_stage_lr, 
                                    momentum=self.args.first_stage_momentum,
                                    weight_decay=self.args.first_stage_weight_decay)
            else:
                opt = torch.optim.Adam(first_stage_params, 
                                    lr=self.args.first_stage_lr,
                                    weight_decay=self.args.first_stage_weight_decay)
            

            # print(325)
            with tqdm(total=self.args.first_stage_epochs * len(train_loader), 
                    desc='First stage training') as pbar:
                # print(328)
                for epoch in range(self.args.first_stage_epochs):
                    for i, data in enumerate(train_loader):
                        if self.args.debug_mode and i > 3:
                            break
                        
                        # LibContinual数据格式适配
                        if isinstance(data, dict):
                            inputs, labels = data['image'], data['label']
                        else:
                            inputs, labels = data[0], data[1]
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device, dtype=torch.long)
                        
                        loss = torch.tensor(0.).to(self.device)
                        
                        opt.zero_grad()
                        
                        # 前向传播
                        clip_logits = self.first_stage(inputs, 
                                                    frozen_past_classes=n_past_classes, 
                                                    cur_classes=n_seen_classes)
                        
                        # 计算CLIP损失 (精确复现)
                        clip_logits[:, :n_past_classes] = -float('inf')
                        loss_clip = loss_fn(clip_logits[:, :n_seen_classes], labels)
                        loss += loss_clip
                        
                        # 正交损失 (精确复现)
                        loss_ortho_coop = self.first_stage.prompter.compute_ortho_loss(
                            frozen_past_classes=n_past_classes, 
                            cur_classes=n_seen_classes
                        )
                        loss += self.args.lambda_ortho_first_stage * loss_ortho_coop
                        
                        # 虚拟批次训练 (精确复现Mammoth逻辑)
                        if i == 0:
                            opt.zero_grad()
                        (loss / self.args.virtual_bs_n).backward()
                        if (i > 0 or self.args.virtual_bs_n == 1) and i % self.args.virtual_bs_n == 0:
                            opt.step()
                            opt.zero_grad()
                        
                        # Wandb日志记录 (可选)
                        # if not self.args.nowand and wandb is not None:
                        #     wandb.log({
                        #         'first_stage_loss': loss.item(),
                        #         'first_stage_lr': opt.param_groups[0]['lr'],
                        #         'first_stage_epoch': epoch,
                        #         'first_stage_loss_clip': loss_clip.item(),
                        #         'first_stage_loss_ortho': loss_ortho_coop.item(),
                        #         'first_stage_iteration': i
                        #     })
                        
                        pbar.update(1)
                        pbar.set_postfix({'loss': loss.item()}, refresh=False)
            
            # 步骤8: 清理优化器 (精确复现)
            opt.zero_grad(set_to_none=True)
            del opt
            torch.cuda.empty_cache()
            
            # 步骤9: 生成重放 (精确复现Mammoth逻辑)
            if self.args.enable_gr:
                self.first_stage.prompter.update_statistics(dataset, current_task)
                self.first_stage.prompter.align(current_task)
            
            # 步骤10: 评估第一阶段性能 (可选，需要实现eval方法)
            # try:
            #     cur_acc = self.eval_first_stage_on_task(dataset, n_seen_classes)
            #     logging.info(f'First stage accuracy: {[acc.item() for acc in cur_acc]}')
            #     logging.info(f'\tAverage: {cur_acc.mean().item():.4f}')
            # except Exception as e:
            #     logging.warning(f"First stage evaluation failed: {e}")
            
        finally:
            # 步骤11: 恢复原始状态 (关键清理步骤)
            if actual_dataset is not None and old_train_transform is not None:
                logging.info("Restoring original transforms")
                actual_dataset.trfms = old_train_transform
            
            # 恢复CLIP模型为float32
            self.first_stage.prompter.clip_model.float()
            self.first_stage.prompter.text_encoder.dtype = torch.float32
            self.first_stage.train(was_training)
        
        print(422)
        logging.info(f"First stage training completed for task {current_task}")

    # @torch.no_grad()
    # def eval_first_stage_on_task(self, dataset, n_seen_classes):
    #     """评估第一阶段性能"""
    #     self.first_stage.eval()
        
    #     # 简化版评估逻辑
    #     total_samples = 0
    #     correct_predictions = 0
        
    #     train_loader = dataset if hasattr(dataset, '__iter__') else dataset.train_loader
        
    #     with torch.no_grad():
    #         for data in train_loader:
    #             if isinstance(data, dict):
    #                 inputs, labels = data['image'], data['label']
    #             else:
    #                 inputs, labels = data[0], data[1]
                
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)
                
    #             clip_logits = self.first_stage(inputs, cur_classes=n_seen_classes)
    #             predictions = torch.argmax(clip_logits, dim=1)
                
    #             correct_predictions += (predictions == labels).sum().item()
    #             total_samples += labels.size(0)
        
    #     accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    #     return torch.tensor([accuracy])  # 返回tensor格式以保持一致性


class STARPrompt(Finetune):
    """
    STAR-Prompt implementation for LibContinual framework
    """

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # STAR-Prompt specific arguments
        self.args = self._setup_args(kwargs)

        # Initialize the STAR-Prompt model
        self.network = STARPromptModel(
            args=self.args,
            backbone=backbone,
            num_classes=num_class,
            device=self.device
        )

        self.network.to(self.device)
        self.task_idx = 0
        self._known_classes = 0
        self.current_task = 0

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.virtual_bs_n = kwargs.get('virtual_bs_n', 1)
        self.accumulated_loss = 0.0
        self.accumulation_count = 0

        # 虚拟批次管理
        self.virtual_batch_count = 0
        self.epoch_iteration = 0
        self.custom_optimizer = None

        logging.info(f"STAR-Prompt initialized with parameters")

    def _setup_args(self, kwargs):
        """Setup arguments for STAR-Prompt from kwargs"""
        from argparse import Namespace

        args = Namespace()

        # First stage arguments
        args.clip_backbone = kwargs.get('clip_backbone', 'ViT-L/14')
        args.first_stage_lr = kwargs.get('first_stage_lr', 0.002)
        args.first_stage_epochs = kwargs.get('first_stage_epochs', 10)
        args.first_stage_optim = kwargs.get('first_stage_optim', 'sgd')
        args.first_stage_momentum = kwargs.get('first_stage_momentum', 0.9)
        args.first_stage_weight_decay = kwargs.get('first_stage_weight_decay', 0.0001)
        args.lambda_ortho_first_stage = kwargs.get('lambda_ortho_first_stage', 0.1)
        args.virtual_bs_n = kwargs.get('virtual_bs_n', 1)

        # Second stage arguments
        args.prompt_mode = kwargs.get('prompt_mode', 'residual')
        args.enable_confidence_modulation = kwargs.get('enable_confidence_modulation', True)
        args.prefix_tuning_prompt_len = kwargs.get('prefix_tuning_prompt_len', 8)
        args.ortho_split_val = kwargs.get('ortho_split_val', 0)
        args.lambda_ortho_second_stage = kwargs.get('lambda_ortho_second_stage', 0.1)
        
        # Ensure prefix_tuning_prompt_len is set for both modes
        if not hasattr(args, 'prefix_tuning_prompt_len'):
            args.prefix_tuning_prompt_len = 8

        # Generative replay arguments
        args.enable_gr = kwargs.get('enable_gr', True)
        args.gr_model = kwargs.get('gr_model', 'mog')
        args.gr_mog_n_components = kwargs.get('gr_mog_n_components', 3)
        args.gr_mog_n_iters_first_stage = kwargs.get('gr_mog_n_iters_first_stage', 100)
        args.gr_mog_n_iters_second_stage = kwargs.get('gr_mog_n_iters_second_stage', 100)
        args.num_samples_gr = kwargs.get('num_samples_gr', 50)
        args.batch_size_gr = kwargs.get('batch_size_gr', 32)
        args.num_epochs_gr_first_stage = kwargs.get('num_epochs_gr_first_stage', 5)
        args.num_epochs_gr_second_stage = kwargs.get('num_epochs_gr_second_stage', 5)
        args.learning_rate_gr_first_stage = kwargs.get('learning_rate_gr_first_stage', 0.01)
        args.learning_rate_gr_second_stage = kwargs.get('learning_rate_gr_second_stage', 0.01)
        args.num_monte_carlo_gr_first_stage = kwargs.get('num_monte_carlo_gr_first_stage', 5)
        args.num_monte_carlo_gr_second_stage = kwargs.get('num_monte_carlo_gr_second_stage', 5)

        # Other arguments
        args.debug_mode = kwargs.get('debug_mode', False)
        args.nowand = kwargs.get('nowand', True)
        args.dataset = kwargs.get('dataset', 'cifar100')
        args.seed = kwargs.get('seed', 1993)
        args.permute_classes = kwargs.get('permute_classes', False)

        # Framework specific
        args.init_cls_num = kwargs.get('init_cls_num', 10)
        args.inc_cls_num = kwargs.get('inc_cls_num', 10)

        return args

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        """
        Called before each task

        这里还是有问题， 修改过
        """
        self.task_idx = task_idx
        self.current_task = task_idx

        torch.cuda.empty_cache()  # ！

        # Update known classes
        if task_idx == 0:
            self._known_classes = self.kwargs['init_cls_num']
        else:
            self._known_classes += self.kwargs['inc_cls_num']

        # Train first stage
        self._train_first_stage(train_loader, task_idx)

        # Update keys for second stage
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if task_idx > 0 else self.kwargs['init_cls_num'])
        self.network.update_keys(n_past_classes, self._known_classes)

        self.network.second_stage.train() # ！ 

        # 关键修复：确保prompter参数可训练（这是Mammoth中的隐式行为）
        self._ensure_prompter_trainable()

        # 关键修改：在这里调用recall_classifier_second_stage
        self.network.recall_classifier_second_stage(task_idx, n_past_classes, self._known_classes)

        # 创建独立的optimizer（完全复现Mammoth）
        second_stage_params = self.get_parameters({})
        
        # 使用与Mammoth相同的设置，这里参数不好传进来
        self.custom_optimizer = torch.optim.Adam(
            second_stage_params, 
            lr=0.001,  # 确保与Mammoth一致
            weight_decay=0,
            betas=(0.9, 0.999)
        )
        
        # print(f"✅ Created independent optimizer with {len(second_stage_params)} parameters")

        # # 但我们需要确保参数设置正确
        # self._setup_second_stage_training() # ！
        # 
    def _ensure_prompter_trainable(self):
        """确保prompter参数可训练 - 复现Mammoth的隐式行为"""
        
        # 激活prompter中的prompt参数
        if hasattr(self.network.second_stage, 'prompter'):
            prompter = self.network.second_stage.prompter
            trainable_count = 0
            
            # 激活所有以 'p_' 或 'a_' 开头的参数（prompt和attention参数）
            for name, param in prompter.named_parameters():
                if name.startswith(('p_', 'a_')) or 'prompt' in name.lower():
                    param.requires_grad = True
                    trainable_count += 1
                    logging.debug(f"Enabled prompter parameter: {name}")
            
            logging.info(f"Enabled {trainable_count} prompter parameters") 


    def _create_dataset_wrapper(self, train_loader):
        """创建数据集包装器"""
        class DatasetWrapper:
            def __init__(self, loader, kwargs):
                self.train_loader = loader
                self.kwargs = kwargs
                self.test_loaders = [loader]
            
            def get_offsets(self, task_id):
                if task_id == 0:
                    return 0, self.kwargs['init_cls_num']
                else:
                    return self.kwargs['init_cls_num'] + (task_id - 1) * self.kwargs['inc_cls_num'], \
                        self.kwargs['init_cls_num'] + task_id * self.kwargs['inc_cls_num']
        
        return DatasetWrapper(train_loader, self.kwargs)


    def _setup_second_stage_training(self):
        """
        设置第二阶段训练的相关配置
        """
        # 确保第二阶段参数可训练
        for name, param in self.network.second_stage.named_parameters():
            param.requires_grad = True
        
        # 确保第一阶段参数冻结
        for name, param in self.network.first_stage.named_parameters():
            param.requires_grad = False
        
        # logging.info("Second stage training setup completed")


    def get_scheduler(self):
        return CosineSchedule(self.opt, K=self.args.n_epochs)

    def _train_first_stage(self, train_loader, task_idx):
        """Train the first stage of STAR-Prompt"""
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if task_idx > 0 else self.kwargs['init_cls_num'])
        
        # 确保第一阶段处于正确状态
        self._setup_first_stage_for_training() # ！ 

        # Create a simple dataset wrapper for training
        class SimpleDataset:
            def __init__(self, loader, kwargs):
                self.loader = loader
                self.train_loader = loader
                self.kwargs = kwargs

            def get_offsets(self, task_id):
                if task_id == 0:
                    return 0, self.kwargs['init_cls_num']
                else:
                    return self.kwargs['init_cls_num'] + (task_id - 1) * self.kwargs['inc_cls_num'], \
                           self.kwargs['init_cls_num'] + task_id * self.kwargs['inc_cls_num']

        dataset = SimpleDataset(train_loader, self.kwargs)

        # Train first stage
        self.network.train_first_stage_on_task(
            dataset=dataset,
            current_task=task_idx,
            n_past_classes=n_past_classes,
            n_seen_classes=self._known_classes,
            loss_fn=self.loss_fn
        )

    def _setup_first_stage_for_training(self):
        """设置第一阶段进行训练"""
        # 首先检查第一阶段是否存在prompt相关模块
        first_stage = self.network.first_stage
        
        # 设置第一阶段为训练模式
        first_stage.train()
        
        # 尝试找到并启用可训练参数
        trainable_count = 0
        for name, param in first_stage.named_parameters():
            # 启用prompt相关参数
            if any(keyword in name.lower() for keyword in ['prompt', 'context', 'learnable', 'adapter']):
                param.requires_grad = True
                trainable_count += 1
                logging.info(f"Enabled training for: {name}")
            # 对于CLIP模型，只训练特定层
            elif 'clip' in name.lower() and any(layer in name.lower() for layer in ['ln_final', 'text_projection', 'logit_scale']):
                param.requires_grad = True
                trainable_count += 1
                logging.info(f"Enabled CLIP parameter for training: {name}")
            else:
                param.requires_grad = False
        
        # 如果没有找到可训练参数，启用一些关键参数
        if trainable_count == 0:
            logging.warning("No obvious trainable parameters found, enabling some parameters...")
            count = 0
            for name, param in first_stage.named_parameters():
                if count < 5:  # 只启用前5个参数作为示例
                    param.requires_grad = True
                    trainable_count += 1
                    count += 1
                    logging.info(f"Force enabled parameter: {name}")
        
        logging.info(f"Total trainable parameters in first stage: {trainable_count}")
        
        if trainable_count == 0:
            raise RuntimeError("No trainable parameters found in first stage!")

    def observe(self, data):
        """Training step"""
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        # 计算类别数量
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if self.task_idx > 0 else self.kwargs['init_cls_num'])

        # Forward pass
        logits = self.network(x, cur_classes=self._known_classes, frozen_past_classes=n_past_classes)

        # ！ 
        with torch.no_grad():  # ！ 
            stream_preds = logits[:, :self._known_classes].argmax(dim=1)
            stream_acc = (stream_preds == y).sum().item() / y.shape[0]

        if n_past_classes > 0:
            logits[:, :n_past_classes] = -float('inf')  # ！ 

        # 只对当前可见的类别计算损失
        loss = self.loss_fn(logits[:, :self._known_classes], y)

        # ！ 

        ortho_loss = self.network.second_stage.prompter.compute_ortho_loss(
            cur_classes=self._known_classes,
            frozen_past_classes=n_past_classes
        )
        loss += self.args.lambda_ortho_second_stage * ortho_loss

        #         # Add orthogonality loss
        # if n_past_classes > 0:
        #     ortho_loss = self.network.second_stage.prompter.compute_ortho_loss(
        #         cur_classes=self._known_classes,
        #         frozen_past_classes=n_past_classes
        #     )
        #     loss += self.args.lambda_ortho_second_stage * ortho_loss

            # 关键修复：检查optimizer状态
        if not hasattr(self, 'custom_optimizer') or self.custom_optimizer is None:
            print("❌ WARNING: custom_optimizer not set, using fallback")
            # 应急方案：直接使用标准训练
            pred = torch.argmax(logits[:, :self._known_classes], dim=1)
            return pred, stream_acc, loss  # 让框架处理
        
        # 检查参数数量
        total_params = sum(len(group['params']) for group in self.custom_optimizer.param_groups)
        if total_params == 0:
            print("❌ CRITICAL: No trainable parameters!")
            pred = torch.argmax(logits[:, :self._known_classes], dim=1)
            return pred, stream_acc, loss

        # 关键修复2：精确复制Mammoth的虚拟批次逻辑
        if self.epoch_iteration == 0:
            if self.custom_optimizer is not None:
                self.custom_optimizer.zero_grad()

        # 精确复现Mammoth的梯度处理
        (loss / self.args.virtual_bs_n).backward()

        # 检查是否需要执行优化步骤（Mammoth的精确条件）
        if (self.epoch_iteration > 0 or self.args.virtual_bs_n == 1) and \
           self.epoch_iteration % self.args.virtual_bs_n == 0:
            if self.custom_optimizer is not None:
                self.custom_optimizer.step()
                self.custom_optimizer.zero_grad()

        # 更新迭代计数
        self.epoch_iteration += 1

        pred = torch.argmax(logits[:, :self._known_classes], dim=1)
        
        # 使用loss的值但断开计算图
        # print(769)

        dummy_loss = torch.tensor(loss.item(), device=self.device, requires_grad=True)
        # print(772)

        return pred, stream_acc, dummy_loss
    
    def set_optimizer(self, optimizer):
        """设置optimizer引用"""
        self.custom_optimizer = optimizer
        total_params = sum(len(group['params']) for group in optimizer.param_groups)
        print(f"✅ STARPrompt: Set optimizer with {total_params} parameters")

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        """Called after each task"""
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if task_idx > 0 else self.kwargs['init_cls_num'])

        # Update statistics for generative replay
        if self.args.enable_gr:
            # Create simple dataset wrapper
            class SimpleDataset:
                def __init__(self, loader, kwargs):
                    self.train_loader = loader
                    self.kwargs = kwargs

                def get_offsets(self, task_id):
                    if task_id == 0:
                        return 0, self.kwargs['init_cls_num']
                    else:
                        return self.kwargs['init_cls_num'] + (task_id - 1) * self.kwargs['inc_cls_num'], \
                               self.kwargs['init_cls_num'] + task_id * self.kwargs['inc_cls_num']

            dataset = SimpleDataset(train_loader, self.kwargs)

            # Update statistics
            self.network.update_statistics(dataset, n_past_classes, self._known_classes)

            # Backup classifier
            self.network.backup(task_idx, n_past_classes, self._known_classes)


             # 对应Mammoth的: if self.current_task > 0:
            if task_idx > 0:
                # 对应Mammoth的: if self.args.seed is not None: torch.manual_seed(self.args.seed)
                if hasattr(self.args, 'seed') and self.args.seed is not None:
                    torch.manual_seed(self.args.seed)
                
                # 对应Mammoth的: self.net.align(self.current_task, self.n_seen_classes, self.loss)
                self.network.align(task_idx,self._known_classes , self.loss_fn)

            # # Alignment step
            # self.network.align(task_idx, self._known_classes, self.loss_fn)

    def inference(self, data):
        """Inference step"""
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            logits = self.network(x, cur_classes=self._known_classes, frozen_past_classes=0)
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == y).item() / x.size(0)

        return pred, acc

    def forward(self, x):
        logits = self.net(x, cur_classes=self._known_classes)
        logits = logits[:, :self._known_classes]
        return logits

    def get_parameters(self, config):
        """
        Get model parameters for LibContinual framework
        按照Mammoth版本，只返回第二阶段的参数
        """
        # 检查是否已经初始化完成
        if not hasattr(self, 'network') or not isinstance(self.network, STARPromptModel):
            # 初始化阶段，返回所有参数
            train_parameters = []
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    train_parameters.append(param)
            return train_parameters
        
        # 正常运行阶段，只返回第二阶段的参数（对应Mammoth的self.net.second_stage.parameters()）
        train_parameters = []
        
        # 只获取第二阶段的参数
        for name, param in self.network.second_stage.named_parameters():
            if param.requires_grad:
                train_parameters.append(param)
        
        # logging.info(f"Training only second stage parameters: {len(train_parameters)}")
        # logging.info(f"First stage parameters are frozen")
        
        return train_parameters

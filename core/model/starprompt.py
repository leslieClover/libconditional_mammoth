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

from core.data.data import CIFARTransform, ImageNetTransform, ImageNetRTransform, TinyImageNetTransform, FiveDatasetsTransform
from core.data.dataset import ContinualDatasets, SingleDataset

try:
    import wandb
except ImportError:
    wandb = None

try:
    import clip
    from clip.model import convert_weights
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")


class LibContinualDatasetAdapter:
    """
    适配器类，用于将LibContinual的ContinualDatasets适配到Mammoth的接口
    """
    def __init__(self, dataset: ContinualDatasets, current_task: int):
        self.train_loader = dataset.get_loader(current_task)
        
        # 创建测试数据集
        test_dataset = ContinualDatasets(
            dataset.dataset, 'test', dataset.task_num, dataset.init_cls_num, 
            dataset.inc_cls_num, dataset.data_root, dataset.cls_map, dataset.trfms, 
            dataset.batchsize, dataset.num_workers, dataset.config
        )
        
        # 获取所有已见任务的测试数据
        self.test_loaders = []
        for t in range(current_task + 1):
            test_loader = test_dataset.get_loader(t)
            if isinstance(test_loader, list):
                self.test_loaders.extend(test_loader)
            else:
                self.test_loaders.append(test_loader)

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
                loss.backward()
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

    @torch.enable_grad()
    def train_first_stage_on_task(self, dataset, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
        """Train the first stage on the current task"""
        return self.first_stage.train_first_stage_on_task(dataset, current_task, n_past_classes, n_seen_classes,
                                                          loss_fn)

    # @torch.enable_grad()
    # def train_first_stage_on_task(self, dataset: ContinualDatasets, current_task: int, n_past_classes: int, n_seen_classes: int, loss_fn):
    #     """
    #     Train the first stage on the current task.
    #     完全按照Mammoth版本适配LibContinual

    #     Args:
    #         dataset: LibContinual的ContinualDatasets对象
    #         current_task: 当前任务索引
    #         n_past_classes: 过去任务的类别数
    #         n_seen_classes: 已见过的总类别数
    #         loss_fn: 损失函数
    #     """
    #     logging.info(f"Starting training of first stage on task: {current_task}")
        
    #     # BEGIN-TASK - 适配LibContinual的数据加载器接口
    #     train_loader = dataset.get_loader(current_task)  # 获取当前任务的训练数据
    #     test_loaders = dataset.get_loader(current_task) if dataset.mode == 'test' else None  # 获取测试数据
        
    #     # 保存原始transform
    #     old_train_transform = train_loader.dataset.trfms
    #     old_test_transform = None
    #     if test_loaders is not None:
    #         if isinstance(test_loaders, list):
    #             old_test_transform = test_loaders[-1].dataset.trfms
    #         else:
    #             old_test_transform = test_loaders.dataset.trfms

    #     # 使用CLIP的预处理
    #     train_loader.dataset.trfms = self.first_stage.prompter.clip_preprocess
    #     if test_loaders is not None:
    #         if isinstance(test_loaders, list):
    #             test_loaders[-1].dataset.trfms = self.first_stage.prompter.clip_preprocess
    #         else:
    #             test_loaders.dataset.trfms = self.first_stage.prompter.clip_preprocess

    #     convert_weights(self.first_stage.prompter.clip_model)  # 转换权重为float16以加速训练
    #     self.first_stage.prompter.text_encoder.dtype = torch.float16
    #     was_training = self.first_stage.training
    #     self.first_stage.train()

    #     first_stage_params = [v for k, v in self.first_stage.named_parameters() if 'prompt_parameters' in k]
    #     if self.args.first_stage_optim == 'sgd':
    #         opt = torch.optim.SGD(first_stage_params, lr=self.args.first_stage_lr, momentum=self.args.first_stage_momentum,
    #                             weight_decay=self.args.first_stage_weight_decay)
    #     else:
    #         opt = torch.optim.Adam(first_stage_params, lr=self.args.first_stage_lr,
    #                             weight_decay=self.args.first_stage_weight_decay)

    #     # MINI TRAINING LOOP FOR CURRENT TASK
    #     with tqdm(total=self.args.first_stage_epochs * len(train_loader), desc='First stage training') as pbar:
    #         for epoch in range(self.args.first_stage_epochs):
    #             for i, data in enumerate(train_loader):
    #                 if self.args.debug_mode and i > 3:
    #                     break
                    
    #                 # 适配LibContinual的数据格式 - 从字典中获取image和label
    #                 inputs = data["image"].to(self.device)
    #                 labels = data["label"].to(self.device, dtype=torch.long)
    #                 loss = torch.tensor(0.).to(self.device)

    #                 opt.zero_grad()
    #                 # Check cur and past classes
    #                 clip_logits = self.first_stage(inputs, frozen_past_classes=n_past_classes, cur_classes=n_seen_classes)

    #                 # compute clip loss
    #                 clip_logits[:, :n_past_classes] = -float('inf')
    #                 loss_clip = loss_fn(clip_logits[:, :n_seen_classes], labels)

    #                 loss += loss_clip

    #                 loss_ortho_coop = self.first_stage.prompter.compute_ortho_loss(frozen_past_classes=n_past_classes, cur_classes=n_seen_classes)
    #                 loss += self.args.lambda_ortho_first_stage * loss_ortho_coop

    #                 if i == 0:
    #                     opt.zero_grad()
    #                 (loss / self.args.virtual_bs_n).backward()
    #                 if (i > 0 or self.args.virtual_bs_n == 1) and i % self.args.virtual_bs_n == 0:
    #                     opt.step()
    #                     opt.zero_grad()

    #                 if not self.args.nowand:
    #                     assert wandb is not None, "wandb is not installed."
    #                     wandb.log({'first_stage_loss': loss.item(),
    #                             'first_stage_lr': opt.param_groups[0]['lr'],
    #                             'first_stage_epoch': epoch,
    #                             'first_stage_loss_clip': loss_clip.item(),
    #                             'first_stage_loss_ortho': loss_ortho_coop.item(),
    #                             'first_stage_iteration': i})

    #                 pbar.update(1)
    #                 pbar.set_postfix({'loss': loss.item()}, refresh=False)

    #     # END-TASK
    #     opt.zero_grad(set_to_none=True)
    #     del opt
    #     torch.cuda.empty_cache()

    #     # Generative replay after end of task
    #     if self.args.enable_gr:
    #         # 为生成式回放创建一个适配器
    #         dataset_adapter = LibContinualDatasetAdapter(dataset, current_task)
    #         self.first_stage.prompter.update_statistics(dataset_adapter, current_task)
    #         self.first_stage.prompter.align(current_task)

    #     # 评估适配
    #     cur_acc = self.eval_first_stage_on_task_libcontinual(dataset, current_task, n_seen_classes)
    #     logging.info(f'First stage accuracy: {[acc.item() for acc in cur_acc]}')
    #     logging.info(f'\tAverage: {cur_acc.mean().item():.4f}')
    #     if not self.args.nowand:
    #         assert wandb is not None, "wandb is not installed."
    #         log_dict = {f'first_stage_acc_{i}': acc.item() for i, acc in enumerate(cur_acc)}
    #         log_dict['first_stage_acc'] = cur_acc.mean().item()
    #         wandb.log(log_dict)

    #     # 恢复原始transforms
    #     train_loader.dataset.trfms = old_train_transform
    #     if test_loaders is not None:
    #         if isinstance(test_loaders, list):
    #             test_loaders[-1].dataset.trfms = old_test_transform
    #         else:
    #             test_loaders.dataset.trfms = old_test_transform

    #     self.first_stage.prompter.clip_model.float()  # 转换回float32
    #     self.first_stage.prompter.text_encoder.dtype = torch.float32
    #     self.first_stage.train(was_training)


    @torch.no_grad()
    def eval_first_stage_on_task_libcontinual(self, dataset: ContinualDatasets, current_task: int, n_seen_classes: int) -> torch.Tensor:
        """
        适配LibContinual的第一阶段评估函数
        计算并返回每个任务的准确率
        """
        was_training = self.first_stage.training
        self.first_stage.eval()
        all_accs = []
        
        # 获取所有已见任务的测试数据
        # 创建一个临时的测试数据集对象来获取测试加载器
        test_dataset = ContinualDatasets(
            dataset.dataset, 'test', dataset.task_num, dataset.init_cls_num, 
            dataset.inc_cls_num, dataset.data_root, dataset.cls_map, dataset.trfms, 
            dataset.batchsize, dataset.num_workers, dataset.config
        )
        
        # 计算总长度用于进度条
        total_batches = 0
        for t in range(current_task + 1):
            test_loader = test_dataset.get_loader(t)
            if isinstance(test_loader, list):
                total_batches += sum(len(loader) for loader in test_loader)
            else:
                total_batches += len(test_loader)
        
        with tqdm(total=total_batches, desc='Eval first stage on seen tasks') as pbar:
            for t in range(current_task + 1):
                test_loader = test_dataset.get_loader(t)
                # 处理LibContinual的test loader可能是单个loader或loader列表的情况
                test_loaders = test_loader if isinstance(test_loader, list) else [test_loader]
                
                for loader in test_loaders:
                    total = 0
                    correct = 0
                    for data in loader:
                        # 适配LibContinual的数据格式
                        inputs = data["image"].to(self.device)
                        labels = data["label"].to(self.device, dtype=torch.long)
                        
                        logits = self.first_stage(inputs, cur_classes=n_seen_classes)[:, :n_seen_classes]
                        _, predicted = torch.max(logits, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        pbar.update(1)
                    
                    all_accs.append(correct / total)
        
        self.first_stage.train(was_training)
        return torch.tensor(all_accs)


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
        args.seed = kwargs.get('seed', 42)
        args.permute_classes = kwargs.get('permute_classes', False)

        # Framework specific
        args.init_cls_num = kwargs.get('init_cls_num', 10)
        args.inc_cls_num = kwargs.get('inc_cls_num', 10)

        return args

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        """Called before each task"""
        self.task_idx = task_idx
        self.current_task = task_idx

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

        # 关键修改：在这里调用recall_classifier_second_stage
        self.network.recall_classifier_second_stage(task_idx, n_past_classes, self._known_classes)
        # # Setup optimizer for second stage
        # self._setup_second_stage_optimizer()

    def _train_first_stage(self, train_loader, task_idx):
        """Train the first stage of STAR-Prompt"""
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if task_idx > 0 else self.kwargs['init_cls_num'])

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

    def observe(self, data):
        """Training step"""
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        n_past_classes = self._known_classes - (
            self.kwargs['inc_cls_num'] if self.task_idx > 0 else self.kwargs['init_cls_num'])

        logits = self.network(x, cur_classes=self._known_classes, frozen_past_classes=n_past_classes)

        # # Compute loss
        # loss = self.loss_fn(logits, y)
        # ⭐ 关键修复: 屏蔽过去类别的logits，防止灾难性遗忘
        # 这个步骤确保模型在当前任务训练时不会"忘记"过去任务的决策边界
        if n_past_classes > 0:
            logits[:, :n_past_classes] = -float('inf')

        # 只对当前可见的类别计算损失
        loss = self.loss_fn(logits[:, :self._known_classes], y)

        # Add orthogonality loss
        if n_past_classes > 0:
            ortho_loss = self.network.second_stage.prompter.compute_ortho_loss(
                cur_classes=self._known_classes,
                frozen_past_classes=n_past_classes
            )
            loss += self.args.lambda_ortho_second_stage * ortho_loss

        # 不在这里执行反向传播 - 让框架处理
        # 框架会调用loss.backward()和optimizer.step()

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item() / x.size(0)

        return pred, acc, loss

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

            # Backup classifier
            self.network.backup(task_idx, n_past_classes, self._known_classes)

            # Update statistics
            self.network.update_statistics(dataset, n_past_classes, self._known_classes)

            # Alignment step
            self.network.align(task_idx, self._known_classes, self.loss_fn)

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

    def get_parameters(self, config):
        """Get model parameters for LibContinual framework"""
        # 返回所有需要训练的参数，让框架创建优化器
        train_parameters = []
        
        # 获取所有需要梯度的参数
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                train_parameters.append(param)
        
        logging.info(f"Total trainable parameters: {len(train_parameters)}")
        
        return train_parameters
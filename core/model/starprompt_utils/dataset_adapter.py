"""
Dataset adapter for LibContinual framework compatibility with mammoth STARPrompt.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Dict, Any
import logging


class LibContinualDatasetAdapter:
    """
    Adapter to make LibContinual dataloaders compatible with mammoth STARPrompt training.
    
    This adapter bridges the gap between LibContinual's data format and mammoth's expected format:
    - LibContinual uses: {'image': tensor, 'label': tensor}
    - mammoth expects: (image_tensor, label_tensor) for test, (aug_image, label, not_aug_image) for train
    """
    
    def __init__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
                 test_loader: Optional[DataLoader] = None, kwargs: Optional[Dict] = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.kwargs = kwargs or {}
        
        # Create test_loaders list for mammoth compatibility
        self.test_loaders = []
        if test_loader is not None:
            self.test_loaders.append(test_loader)
            
        # mammoth expects these attributes
        self.args = self._create_args_namespace()
        
        # Task and class information
        self.N_CLASSES_PER_TASK = self.kwargs.get('inc_cls_num', 10)
        self.N_TASKS = self.kwargs.get('task_num', 10)
        self.N_CLASSES = self.kwargs.get('total_cls_num', 100)
        self.SETTING = 'class-il'  # LibContinual typically uses class-incremental
        
        # Current task tracking
        self.c_task = -1  # Will be incremented in get_offsets
        
        logging.info(f"Dataset adapter initialized: {self.N_TASKS} tasks, {self.N_CLASSES_PER_TASK} classes per task")
    
    def _create_args_namespace(self):
        """Create args namespace for mammoth compatibility"""
        from argparse import Namespace
        
        args = Namespace()
        args.validation_mode = 'current'  # or 'complete'
        args.validation = False
        args.permute_classes = self.kwargs.get('permute_classes', False)
        args.noise_rate = 0
        args.seed = self.kwargs.get('seed', 42)
        args.batch_size = self.kwargs.get('batch_size', 32)
        args.debug_mode = self.kwargs.get('debug_mode', False)
        
        if args.permute_classes:
            # Create class order for permutation
            args.class_order = np.arange(self.N_CLASSES)
            if args.seed is not None:
                np.random.seed(args.seed)
                np.random.shuffle(args.class_order)
        
        return args
    
    def get_offsets(self, task_idx: Optional[int] = None) -> Tuple[int, int]:
        """
        Get class offsets for a given task (mammoth compatibility).
        
        Args:
            task_idx: Task index. If None, uses current task.
            
        Returns:
            Tuple of (start_class, end_class)
        """
        if task_idx is None:
            task_idx = self.c_task
            
        if task_idx == 0:
            start_c = 0
            end_c = self.kwargs.get('init_cls_num', 10)
        else:
            init_cls = self.kwargs.get('init_cls_num', 10)
            inc_cls = self.kwargs.get('inc_cls_num', 10)
            start_c = init_cls + (task_idx - 1) * inc_cls
            end_c = init_cls + task_idx * inc_cls
            
        return start_c, end_c
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Return current train and test loaders (mammoth compatibility).
        This method is called by mammoth's training loop.
        """
        # Increment task counter (mammoth behavior)
        self.c_task += 1
        
        # Wrap loaders to convert data format
        wrapped_train_loader = self._wrap_loader(self.train_loader, is_train=True)
        wrapped_test_loader = self._wrap_loader(self.test_loader, is_train=False)
        
        # Update test_loaders list
        if wrapped_test_loader is not None:
            self.test_loaders.append(wrapped_test_loader)
            
        return wrapped_train_loader, wrapped_test_loader
    
    def _wrap_loader(self, loader: Optional[DataLoader], is_train: bool = True) -> Optional[DataLoader]:
        """
        Wrap a LibContinual loader to convert data format to mammoth format.
        """
        if loader is None:
            return None
            
        # Create a new DataLoader with format conversion
        dataset = DataFormatConverter(loader.dataset, is_train=is_train)
        
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=loader.shuffle if hasattr(loader, 'shuffle') else is_train,
            num_workers=getattr(loader, 'num_workers', 0),
            pin_memory=getattr(loader, 'pin_memory', False),
            drop_last=getattr(loader, 'drop_last', False),
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to handle data format conversion"""
        try:
            # 分离图像和标签
            if isinstance(batch[0], dict):
                # LibContinual 格式
                images = torch.stack([item['image'] for item in batch])
                labels = torch.stack([item['label'] for item in batch])
            elif isinstance(batch[0], (tuple, list)):
                # mammoth 格式
                images = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
            else:
                raise ValueError(f"不支持的批次数据格式: {type(batch[0])}")
            
            # 确保标签是长整型
            if labels.dtype != torch.long:
                labels = labels.long()
                
            # 返回适合的格式
            if self.is_train:
                # 训练时返回 (aug_image, label, not_aug_image) 格式
                return (images, labels, images)
            else:
                # 测试时返回 (image, label) 格式
                return (images, labels)
                
        except Exception as e:
            logging.error(f"数据整理(collate)时出错: {str(e)}")
            logging.error(f"批次大小: {len(batch)}")
            logging.error(f"批次第一个元素类型: {type(batch[0])}")
            raise


class DataFormatConverter:
    """
    Converts LibContinual dataset format to mammoth format.
    """
    
    def __init__(self, dataset, is_train: bool = True):
        self.dataset = dataset
        self.is_train = is_train
        
        # Required attributes for mammoth compatibility
        self.data = self._extract_data()
        self.targets = self._extract_targets()
        
    def _extract_data(self):
        """Extract data array from LibContinual dataset"""
        try:
            # Try to get all data at once
            all_data = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if isinstance(item, dict):
                    all_data.append(item['image'])
                else:
                    all_data.append(item[0])  # Assume first element is image
            return torch.stack(all_data) if all_data else np.array([])
        except:
            # Fallback: return empty array, will be populated on demand
            return np.array([])
    
    def _extract_targets(self):
        """Extract targets array from LibContinual dataset"""
        try:
            all_targets = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if isinstance(item, dict):
                    all_targets.append(item['label'])
                else:
                    all_targets.append(item[1])  # Assume second element is label
            return np.array(all_targets)
        except:
            return np.array([])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Get item in mammoth format.
        """
        item = self.dataset[index]
        
        # Convert from LibContinual format to mammoth format
        if isinstance(item, dict):
            image = item['image']
            label = item['label']
        else:
            # Assume tuple format (image, label, ...)
            image, label = item[0], item[1]
        
        if self.is_train:
            # Training format: (aug_image, label, not_aug_image)
            # Since LibContinual doesn't separate aug/not_aug, we use the same image
            not_aug_image = image.clone() if isinstance(image, torch.Tensor) else image
            return image, label, not_aug_image
        else:
            # Test format: (image, label)
            return image, label


# Utility function to create adapter from LibContinual components
def create_dataset_adapter(train_loader: DataLoader, test_loaders: Dict[int, DataLoader], 
                          task_idx: int, kwargs: Dict) -> LibContinualDatasetAdapter:
    """
    Create dataset adapter for a specific task.
    
    Args:
        train_loader: Training data loader
        test_loaders: Dictionary of test loaders {task_id: loader}
        task_idx: Current task index
        kwargs: Configuration parameters
        
    Returns:
        LibContinualDatasetAdapter instance
    """
    test_loader = test_loaders.get(task_idx, None)
    
    adapter = LibContinualDatasetAdapter(
        train_loader=train_loader,
        test_loader=test_loader,
        kwargs=kwargs
    )
    
    return adapter
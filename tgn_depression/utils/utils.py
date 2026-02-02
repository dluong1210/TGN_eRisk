"""
Utility functions for TGN Depression Detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def get_device_for_rank(local_rank: int, gpu_ids: list) -> torch.device:
    """
    Get device for current process in multi-GPU (DDP) training.
    Use when each process has CUDA_VISIBLE_DEVICES set to a single GPU;
    then cuda:0 is the correct device for this process.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    # When using spawn with gpu_ids, we set CUDA_VISIBLE_DEVICES per process,
    # so the only visible GPU is cuda:0 in this process.
    return torch.device('cuda:0')


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: Array of labels (0 or 1)
    
    Returns:
        Class weights tensor
    """
    n_total = len(labels)
    n_pos = np.sum(labels)
    n_neg = n_total - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])
    
    # Weight inversely proportional to class frequency
    weight_pos = n_total / (2 * n_pos)
    weight_neg = n_total / (2 * n_neg)
    
    return torch.tensor([weight_neg, weight_pos])


class EarlyStopMonitor:
    """
    Monitor for early stopping during training.
    """
    
    def __init__(self, 
                 max_round: int = 5, 
                 higher_better: bool = True,
                 tolerance: float = 1e-6):
        """
        Args:
            max_round: Maximum rounds without improvement before stopping
            higher_better: If True, higher metric is better
            tolerance: Minimum improvement to count as better
        """
        self.max_round = max_round
        self.higher_better = higher_better
        self.tolerance = tolerance
        
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
    
    def early_stop_check(self, curr_val: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            curr_val: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if not self.higher_better:
            curr_val = -curr_val
        
        if self.last_best is None:
            self.last_best = curr_val
            self.best_epoch = self.epoch_count
        elif (curr_val - self.last_best) / (abs(self.last_best) + 1e-10) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        
        self.epoch_count += 1
        
        return self.num_round >= self.max_round


class MergeLayer(nn.Module):
    """
    MLP layer that merges two input vectors.
    Used for computing edge probabilities or combining features.
    """
    
    def __init__(self, dim1: int, dim2: int, dim3: int, dim4: int):
        """
        Args:
            dim1: First input dimension
            dim2: Second input dimension
            dim3: Hidden dimension
            dim4: Output dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x1: First input [batch, dim1]
            x2: Second input [batch, dim2]
        
        Returns:
            Output [batch, dim4]
        """
        x = torch.cat([x1, x2], dim=-1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ClassificationHead(nn.Module):
    """
    Classification head for depression detection.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node embeddings [batch, input_dim]
        
        Returns:
            Logits [batch, num_classes]
        """
        return self.classifier(x)


def compute_time_statistics(
    sources: np.ndarray,
    destinations: np.ndarray, 
    timestamps: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute time statistics for normalization.
    
    Args:
        sources: Source node ids
        destinations: Destination node ids
        timestamps: Timestamps
    
    Returns:
        mean_time_shift_src, std_time_shift_src,
        mean_time_shift_dst, std_time_shift_dst
    """
    last_timestamp_sources = {}
    last_timestamp_dst = {}
    all_timediffs_src = []
    all_timediffs_dst = []
    
    for source_id, dest_id, timestamp in zip(sources, destinations, timestamps):
        if source_id not in last_timestamp_sources:
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst:
            last_timestamp_dst[dest_id] = 0
        
        all_timediffs_src.append(timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(timestamp - last_timestamp_dst[dest_id])
        
        last_timestamp_sources[source_id] = timestamp
        last_timestamp_dst[dest_id] = timestamp
    
    mean_time_shift_src = np.mean(all_timediffs_src) if all_timediffs_src else 0
    std_time_shift_src = np.std(all_timediffs_src) if all_timediffs_src else 1
    mean_time_shift_dst = np.mean(all_timediffs_dst) if all_timediffs_dst else 0
    std_time_shift_dst = np.std(all_timediffs_dst) if all_timediffs_dst else 1
    
    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

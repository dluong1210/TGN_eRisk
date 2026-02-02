"""
Message Aggregator Module for TGN.

Aggregate multiple messages for the same node into a single message.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class MessageAggregator(nn.Module):
    """
    Base class for message aggregators.
    
    When multiple events involve the same node in a batch,
    this module aggregates their messages.
    """
    
    def __init__(self, device: torch.device):
        super(MessageAggregator, self).__init__()
        self.device = device
    
    def aggregate(
        self, 
        node_ids: List[int],
        messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Aggregate messages for each node.
        
        Args:
            node_ids: List of node ids to aggregate
            messages: Dict mapping node_id to list of (message, timestamp) tuples
        
        Returns:
            unique_node_ids: List of nodes that have messages
            aggregated_messages: Tensor [n_unique, message_dim]
            timestamps: Tensor [n_unique]
        """
        raise NotImplementedError


class LastMessageAggregator(MessageAggregator):
    """
    Keep only the most recent message for each node.
    
    Fast and simple - good default choice.
    """
    
    def __init__(self, device: torch.device):
        super(LastMessageAggregator, self).__init__(device)
    
    def aggregate(
        self,
        node_ids: List[int],
        messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Keep only the last message for each node."""
        
        unique_node_ids = np.unique(node_ids)
        to_update_node_ids = []
        unique_messages = []
        unique_timestamps = []
        
        for node_id in unique_node_ids:
            node_id = int(node_id)
            if node_id in messages and len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                # Get last message (most recent)
                last_message, last_timestamp = messages[node_id][-1]
                unique_messages.append(last_message)
                unique_timestamps.append(last_timestamp)
        
        if len(to_update_node_ids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = torch.tensor([]).to(self.device)
            unique_timestamps = torch.tensor([]).to(self.device)
        
        return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    """
    Average all messages for each node.
    
    Preserves more information but slower.
    """
    
    def __init__(self, device: torch.device):
        super(MeanMessageAggregator, self).__init__(device)
    
    def aggregate(
        self,
        node_ids: List[int],
        messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Average all messages for each node."""
        
        unique_node_ids = np.unique(node_ids)
        to_update_node_ids = []
        unique_messages = []
        unique_timestamps = []
        
        for node_id in unique_node_ids:
            node_id = int(node_id)
            if node_id in messages and len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                
                # Stack all messages and average
                node_messages = torch.stack([m[0] for m in messages[node_id]])
                mean_message = torch.mean(node_messages, dim=0)
                unique_messages.append(mean_message)
                
                # Use timestamp of last message
                unique_timestamps.append(messages[node_id][-1][1])
        
        if len(to_update_node_ids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = torch.tensor([]).to(self.device)
            unique_timestamps = torch.tensor([]).to(self.device)
        
        return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(
    aggregator_type: str,
    device: torch.device
) -> MessageAggregator:
    """
    Factory function to create message aggregator.
    
    Args:
        aggregator_type: 'last' or 'mean'
        device: Device to use
    
    Returns:
        MessageAggregator instance
    """
    if aggregator_type == 'last':
        return LastMessageAggregator(device=device)
    elif aggregator_type == 'mean':
        return MeanMessageAggregator(device=device)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")

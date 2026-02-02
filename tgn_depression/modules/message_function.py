"""
Message Function Module for TGN.

Tạo messages từ các raw inputs (memory, edge features, time).
"""

import torch
import torch.nn as nn
from typing import Optional


class MessageFunction(nn.Module):
    """
    Base class for message functions.
    
    Message function transforms raw inputs into messages
    that will be used to update memory.
    """
    
    def __init__(self):
        super(MessageFunction, self).__init__()
    
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """
        Compute messages from raw inputs.
        
        Args:
            raw_messages: Raw message tensor [batch, raw_dim]
        
        Returns:
            Processed messages [batch, message_dim]
        """
        raise NotImplementedError


class IdentityMessageFunction(MessageFunction):
    """
    Identity message function - returns input unchanged.
    
    raw_message = [src_memory || dst_memory || edge_features || time_encoding]
    """
    
    def __init__(self):
        super(IdentityMessageFunction, self).__init__()
    
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """Return raw messages unchanged."""
        return raw_messages


class MLPMessageFunction(MessageFunction):
    """
    MLP message function - transforms raw messages through MLP.
    """
    
    def __init__(self, 
                 raw_message_dimension: int,
                 message_dimension: int):
        """
        Args:
            raw_message_dimension: Input dimension
            message_dimension: Output dimension
        """
        super(MLPMessageFunction, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension)
        )
    
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """Transform raw messages through MLP."""
        return self.mlp(raw_messages)


def get_message_function(
    module_type: str,
    raw_message_dimension: int,
    message_dimension: int
) -> MessageFunction:
    """
    Factory function to create message function.
    
    Args:
        module_type: 'identity' or 'mlp'
        raw_message_dimension: Input dimension
        message_dimension: Output dimension
    
    Returns:
        MessageFunction instance
    """
    if module_type == 'identity':
        return IdentityMessageFunction()
    elif module_type == 'mlp':
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    else:
        raise ValueError(f"Unknown message function type: {module_type}")

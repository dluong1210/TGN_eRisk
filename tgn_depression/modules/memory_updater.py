"""
Memory Updater Module for TGN.

Update node memory using aggregated messages.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

try:
    from .memory import Memory
except ImportError:
    from memory import Memory


class MemoryUpdater(nn.Module):
    """
    Base class for memory updaters.
    
    Updates node memory vectors using messages.
    """
    
    def __init__(self):
        super(MemoryUpdater, self).__init__()
    
    def update_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """
        Update memory for specified nodes.
        
        Args:
            unique_node_ids: List of node ids to update
            unique_messages: Messages tensor [n_nodes, message_dim]
            timestamps: Timestamps tensor [n_nodes]
        """
        raise NotImplementedError
    
    def get_updated_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get updated memory without persisting.
        
        Returns:
            Updated memory tensor, updated last_update tensor
        """
        raise NotImplementedError


class SequenceMemoryUpdater(MemoryUpdater):
    """
    Base class for sequence-based memory updaters (GRU, RNN, LSTM).
    """
    
    def __init__(self,
                 memory: Memory,
                 message_dimension: int,
                 memory_dimension: int,
                 device: torch.device):
        """
        Args:
            memory: Memory module to update
            message_dimension: Dimension of messages
            memory_dimension: Dimension of memory
            device: Device to use
        """
        super(SequenceMemoryUpdater, self).__init__()
        
        self.memory = memory
        self.message_dimension = message_dimension
        self.memory_dimension = memory_dimension
        self.device = device
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(memory_dimension)
        
        # To be defined in subclass
        self.memory_updater = None
    
    def update_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Update memory in-place."""
        if len(unique_node_ids) == 0:
            return
        
        # Verify temporal consistency
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to time in the past"
        
        # Get current memory
        current_memory = self.memory.get_memory(unique_node_ids)
        
        # Update using RNN/GRU
        updated_memory = self.memory_updater(unique_messages, current_memory)
        
        # Apply layer norm
        # updated_memory = self.layer_norm(updated_memory)
        
        # Store updated memory
        self.memory.set_memory(unique_node_ids, updated_memory)
        self.memory.last_update[unique_node_ids] = timestamps
    
    def get_updated_memory(
        self,
        unique_node_ids: List[int],
        unique_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get updated memory without persisting."""
        if len(unique_node_ids) == 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        
        # Verify temporal consistency
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to time in the past"
        
        # Clone memory
        updated_memory = self.memory.memory.data.clone()
        updated_last_update = self.memory.last_update.data.clone()
        
        # Update specified nodes
        current_memory = updated_memory[unique_node_ids]
        new_memory = self.memory_updater(unique_messages, current_memory)
        updated_memory[unique_node_ids] = new_memory
        updated_last_update[unique_node_ids] = timestamps
        
        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    """
    GRU-based memory updater.
    
    memory_new = GRU(message, memory_old)
    """
    
    def __init__(self,
                 memory: Memory,
                 message_dimension: int,
                 memory_dimension: int,
                 device: torch.device):
        super(GRUMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )


class RNNMemoryUpdater(SequenceMemoryUpdater):
    """
    Vanilla RNN-based memory updater.
    """
    
    def __init__(self,
                 memory: Memory,
                 message_dimension: int,
                 memory_dimension: int,
                 device: torch.device):
        super(RNNMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        
        self.memory_updater = nn.RNNCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )


def get_memory_updater(
    module_type: str,
    memory: Memory,
    message_dimension: int,
    memory_dimension: int,
    device: torch.device
) -> MemoryUpdater:
    """
    Factory function to create memory updater.
    
    Args:
        module_type: 'gru' or 'rnn'
        memory: Memory module
        message_dimension: Dimension of messages
        memory_dimension: Dimension of memory
        device: Device to use
    
    Returns:
        MemoryUpdater instance
    """
    if module_type == 'gru':
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == 'rnn':
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    else:
        raise ValueError(f"Unknown memory updater type: {module_type}")

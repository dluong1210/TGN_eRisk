"""
Memory Module for TGN.

Memory lưu trữ trạng thái của mỗi node (user) và được update
sau mỗi interaction.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class Memory(nn.Module):
    """
    Node memory module.
    
    Mỗi node có một memory vector được update theo thời gian
    thông qua các interactions.
    """
    
    def __init__(self,
                 n_nodes: int,
                 memory_dimension: int,
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            n_nodes: Total number of nodes (users)
            memory_dimension: Dimension of memory vectors
            device: Device to store memory on
        """
        super(Memory, self).__init__()
        
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        
        # Initialize memory
        self.__init_memory__()
    
    def __init_memory__(self):
        """
        Initialize memory to zeros.
        Called at the start of each epoch or when processing new conversation.
        """
        # Memory vectors for each node
        self.memory = nn.Parameter(
            torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
            requires_grad=False
        )
        
        # Last update timestamp for each node
        self.last_update = nn.Parameter(
            torch.zeros(self.n_nodes).to(self.device),
            requires_grad=False
        )
        
        # Raw messages waiting to be processed
        # Format: {node_id: [(message_tensor, timestamp), ...]}
        self.messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    
    def get_memory(self, node_idxs) -> torch.Tensor:
        """
        Get memory vectors for specified nodes.
        
        Args:
            node_idxs: Node indices (int, list, or array)
        
        Returns:
            Memory tensor [n_nodes, memory_dim]
        """
        return self.memory[node_idxs, :]
    
    def set_memory(self, node_idxs, values: torch.Tensor):
        """
        Set memory values for specified nodes.
        
        Args:
            node_idxs: Node indices
            values: New memory values
        """
        self.memory[node_idxs, :] = values
    
    def get_last_update(self, node_idxs) -> torch.Tensor:
        """Get last update timestamps for nodes."""
        return self.last_update[node_idxs]
    
    def store_raw_messages(self, 
                           nodes: List[int], 
                           node_id_to_messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]):
        """
        Store raw messages for later processing.
        
        Args:
            nodes: List of node ids
            node_id_to_messages: Mapping from node id to list of (message, timestamp) tuples
        """
        for node in nodes:
            if node in node_id_to_messages:
                self.messages[node].extend(node_id_to_messages[node])
    
    def get_messages(self, node_idxs: List[int]) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get stored messages for specified nodes.
        
        Args:
            node_idxs: Node indices to get messages for
        
        Returns:
            Dictionary mapping node id to list of messages
        """
        return {node: self.messages.get(node, []) for node in node_idxs}
    
    def clear_messages(self, nodes: List[int]):
        """
        Clear stored messages for specified nodes.
        
        Args:
            nodes: Node ids to clear messages for
        """
        for node in nodes:
            self.messages[node] = []
    
    def backup_memory(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Backup current memory state.
        
        Returns:
            Tuple of (memory_data, last_update_data, messages_clone)
        """
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        
        return (
            self.memory.data.clone(),
            self.last_update.data.clone(),
            messages_clone
        )
    
    def restore_memory(self, memory_backup: Tuple[torch.Tensor, torch.Tensor, Dict]):
        """
        Restore memory from backup.
        
        Args:
            memory_backup: Tuple from backup_memory()
        """
        self.memory.data = memory_backup[0].clone()
        self.last_update.data = memory_backup[1].clone()
        
        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
    
    def detach_memory(self):
        """
        Detach memory from computation graph.
        Called after backprop to prevent gradient accumulation.
        """
        self.memory.detach_()
        
        # Detach all stored messages
        for k, v in self.messages.items():
            new_messages = []
            for message, timestamp in v:
                new_messages.append((message.detach(), timestamp))
            self.messages[k] = new_messages
    
    def reset_state(self):
        """Reset memory state (alias for __init_memory__)."""
        self.__init_memory__()

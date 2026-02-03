"""
Node memory module.

Chỉ lưu memory cho các node đã được cập nhật (node trong L-hop ego),
không cấp phát full buffer (n_nodes, dim) → giảm VRAM khi n_nodes lớn.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import numpy as np


class Memory(nn.Module):
    """
    Node memory module — sparse: chỉ lưu các node đã update (ego), không full graph.
    """

    def __init__(self,
                 n_nodes: int,
                 memory_dimension: int,
                 device: torch.device = torch.device('cpu')):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        # Sparse: chỉ lưu node đã được update (trong ego), không cấp phát (n_nodes, dim)
        self._memory: Dict[int, torch.Tensor] = {}
        self._last_update: Dict[int, torch.Tensor] = {}
        self.messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

    @property
    def memory(self) -> torch.Tensor:
        """Compat: code cũ có thể đọc .memory; trả về tensor ảo (0, dim) để không vỡ."""
        if not hasattr(self, '_dummy_memory') or self._dummy_memory.shape[1] != self.memory_dimension:
            self.register_buffer('_dummy_memory', torch.zeros(0, self.memory_dimension, device=self.device))
        return self._dummy_memory

    def _zero_row(self) -> torch.Tensor:
        return torch.zeros(1, self.memory_dimension, device=self.device, dtype=torch.float32).squeeze(0)

    def get_memory(self, node_idxs: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Trả về [len(node_idxs), memory_dim]; node chưa có thì trả về 0."""
        if hasattr(node_idxs, '__len__') and len(node_idxs) == 0:
            return torch.zeros(0, self.memory_dimension, device=self.device, dtype=torch.float32)
        idx = np.asarray(node_idxs).flatten()
        out = torch.zeros(len(idx), self.memory_dimension, device=self.device, dtype=torch.float32)
        for i in range(len(idx)):
            nid = int(idx[i])
            if nid in self._memory:
                out[i] = self._memory[nid]
        return out

    def set_memory(self, node_idxs: Union[List[int], np.ndarray], values: torch.Tensor):
        """Chỉ lưu các hàng tương ứng node_idxs."""
        idx = np.asarray(node_idxs).flatten()
        for i in range(len(idx)):
            nid = int(idx[i])
            self._memory[nid] = values[i]

    def get_last_update(self, node_idxs: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Trả về last_update cho từng node; chưa có thì 0."""
        idx = np.asarray(node_idxs).flatten()
        out = torch.zeros(len(idx), device=self.device, dtype=torch.float32)
        for i in range(len(idx)):
            nid = int(idx[i])
            if nid in self._last_update:
                t = self._last_update[nid]
                out[i] = t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else (t if isinstance(t, float) else t.item())
        return out

    def set_last_update(self, node_idxs: Union[List[int], np.ndarray], timestamps: torch.Tensor):
        """Ghi last_update chỉ cho các node_idxs (ego)."""
        idx = np.asarray(node_idxs).flatten()
        for i in range(len(idx)):
            nid = int(idx[i])
            self._last_update[nid] = timestamps[i] if timestamps.dim() > 0 else timestamps

    def store_raw_messages(self,
                           nodes: List[int],
                           node_id_to_messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]):
        for node in nodes:
            if node in node_id_to_messages:
                self.messages[node].extend(node_id_to_messages[node])

    def get_messages(self, node_idxs: List[int]) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        return {node: self.messages.get(node, []) for node in node_idxs}

    def clear_messages(self, nodes: List[int]):
        for node in nodes:
            self.messages[node] = []

    def backup_memory(self) -> Tuple[Dict, Dict, Dict]:
        """Backup sparse state."""
        mem_clone = {k: v.clone() for k, v in self._memory.items()}
        last_clone = {k: v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in self._last_update.items()}
        msg_clone = {}
        for k, v in self.messages.items():
            msg_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return (mem_clone, last_clone, msg_clone)

    def restore_memory(self, backup: Tuple[Dict, Dict, Dict]):
        self._memory = {k: v.clone() for k, v in backup[0].items()}
        self._last_update = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in backup[1].items()}
        self.messages = defaultdict(list)
        for k, v in backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self._memory = {k: v.detach() for k, v in self._memory.items()}
        self._last_update = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in self._last_update.items()}
        self.messages = defaultdict(list)

    def reset_state(self):
        self.__init_memory__()



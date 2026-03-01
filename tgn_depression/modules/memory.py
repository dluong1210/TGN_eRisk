"""
Node memory module.

Uses a full buffer [n_nodes, dim] so that gradients flow from the classification
loss back to the memory updater (GRU). Updates use truncated BPTT:
buffer = buffer.detach().clone(); buffer[ids] = values.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import numpy as np


class Memory(nn.Module):
    """
    Node memory module — full buffer so embedding(memory) is differentiable.
    TGN gốc: loss -> embedding(memory) -> memory -> memory_updater (GRU).
    """

    def __init__(self,
                 n_nodes: int,
                 memory_dimension: int,
                 device: torch.device = torch.device('cpu')):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.register_buffer(
            "memory_buffer",
            torch.zeros(n_nodes, memory_dimension, device=device, dtype=torch.float32),
        )
        self.__init_memory__()

    def __init_memory__(self):
        self._last_update: Dict[int, torch.Tensor] = {}
        self.messages: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

    @property
    def memory(self) -> torch.Tensor:
        """Compat: code cũ có thể đọc .memory; trả về buffer."""
        return self.memory_buffer

    def _zero_row(self) -> torch.Tensor:
        return torch.zeros(1, self.memory_dimension, device=self.device, dtype=torch.float32).squeeze(0)

    def get_memory(self, node_idxs: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Trả về [len(node_idxs), memory_dim]; slice từ buffer để giữ computation graph."""
        if hasattr(node_idxs, '__len__') and len(node_idxs) == 0:
            return torch.zeros(0, self.memory_dimension, device=self.device, dtype=torch.float32)
        idx = torch.as_tensor(
            np.asarray(node_idxs).flatten(), device=self.memory_buffer.device, dtype=torch.long
        )
        return self.memory_buffer[idx]

    def set_memory(self, node_idxs: Union[List[int], np.ndarray], values: torch.Tensor):
        """
        Cập nhật buffer có đạo hàm: new = old.detach() * (1 - mask) + scatter(values at ids).
        Truncated BPTT: old bị detach; gradient chỉ chảy qua values (GRU output).
        """
        idx = torch.as_tensor(
            np.asarray(node_idxs).flatten(), device=self.memory_buffer.device, dtype=torch.long
        )
        old = self.memory_buffer.detach().clone()
        # scatter: tensor bằng 0 ngoài ids, bằng values tại ids
        scatter = torch.zeros_like(old)
        scatter[idx] = values
        # mask: 1 tại ids để thay thế
        mask = torch.zeros(old.shape[0], 1, device=old.device, dtype=old.dtype)
        mask[idx] = 1.0
        self.memory_buffer = old * (1.0 - mask) + scatter

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

    def backup_memory(self) -> Tuple[torch.Tensor, Dict, Dict]:
        """Backup buffer and ephemeral state."""
        mem_clone = self.memory_buffer.clone()
        last_clone = {
            k: v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
            for k, v in self._last_update.items()
        }
        msg_clone = {}
        for k, v in self.messages.items():
            msg_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return (mem_clone, last_clone, msg_clone)

    def restore_memory(self, backup: Tuple[torch.Tensor, Dict, Dict]):
        self.memory_buffer.copy_(backup[0])
        self._last_update = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in backup[1].items()
        }
        self.messages = defaultdict(list)
        for k, v in backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        if self.memory_buffer.requires_grad:
            self.memory_buffer.detach_()
        self._last_update = {
            k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in self._last_update.items()
        }
        self.messages = defaultdict(list)

    def reset_state(self):
        # Tạo buffer mới hoàn toàn, KHÔNG tham chiếu buffer cũ. Sau backward() PyTorch
        # có thể đã free buffer cũ; dùng zeros_like/detach vẫn truy cập nó → lỗi.
        dev = self.memory_buffer.device  # chỉ đọc .device, không đụng storage
        self.memory_buffer = torch.zeros(
            self.n_nodes, self.memory_dimension,
            device=dev, dtype=torch.float32,
        )
        self.__init_memory__()

    def get_full_memory_tensor(self) -> torch.Tensor:
        """Return the full memory buffer for embedding module (same tensor, gradient flows)."""
        return self.memory_buffer

    def free_nodes_except(self, keep_node: int):
        """
        Giữ memory của keep_node, zero các node khác (sau mỗi conversation).
        Dùng mask nhân (không detach) để gradient vẫn chảy qua keep_node → model có thể học.
        Trước đây detach khiến buffer mất grad → embedding không có gradient khi conv cuối
        không update target_user, hoặc gradient bị cắt quá sớm.
        """
        mask = torch.zeros(
            self.n_nodes, 1, device=self.memory_buffer.device, dtype=self.memory_buffer.dtype
        )
        mask[keep_node, 0] = 1.0
        self.memory_buffer = self.memory_buffer * mask
        nodes_to_remove = [n for n in list(self._last_update.keys()) if n != keep_node]
        for n in nodes_to_remove:
            del self._last_update[n]
        self.messages.clear()
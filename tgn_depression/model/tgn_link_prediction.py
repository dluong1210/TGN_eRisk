"""
TGN cho link prediction (pretrain) — giao diện giống tgn/model/tgn.py.

- compute_temporal_embeddings(sources, destinations, negatives, edge_times, edge_idxs)
- compute_edge_probabilities(...) -> pos_scores, neg_scores (sigmoid)
- Memory update at start hoặc end of batch; affinity_score = MergeLayer.
Dùng modules của tgn_depression (memory, message, aggregator, updater, embedding).
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Optional, Tuple

try:
    from ..modules.memory import Memory
    from ..modules.message_function import get_message_function
    from ..modules.message_aggregator import get_message_aggregator
    from ..modules.memory_updater import get_memory_updater
    from ..modules.embedding_module import TimeEncode, get_embedding_module
    from ..utils.utils import MergeLayer
    from ..utils.neighbor_finder import NeighborFinder, get_neighbor_finder
except ImportError:
    from modules.memory import Memory
    from modules.message_function import get_message_function
    from modules.message_aggregator import get_message_aggregator
    from modules.memory_updater import get_memory_updater
    from modules.embedding_module import TimeEncode, get_embedding_module
    from utils.utils import MergeLayer
    from utils.neighbor_finder import NeighborFinder, get_neighbor_finder


class TGNLinkPrediction(nn.Module):
    """
    TGN encoder + affinity score cho link prediction (pretrain).
    API giống tgn/model/tgn.py: compute_temporal_embeddings, compute_edge_probabilities.
    """

    def __init__(
        self,
        neighbor_finder: NeighborFinder,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        device: torch.device,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
        use_memory: bool = True,
        memory_update_at_start: bool = True,
        message_dimension: int = 100,
        memory_dimension: int = 172,
        embedding_module_type: str = "graph_attention",
        message_function_type: str = "identity",
        aggregator_type: str = "last",
        memory_updater_type: str = "gru",
        n_neighbors: Optional[int] = 10,
    ):
        super().__init__()
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors or 10
        self.use_memory = use_memory
        self.memory_update_at_start = memory_update_at_start

        node_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        edge_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.node_raw_features = node_features
        self.edge_raw_features = edge_features
        self.n_nodes = node_features.shape[0]
        self.n_node_features = node_features.shape[1]
        self.n_edge_features = edge_features.shape[1]
        self.embedding_dimension = self.n_node_features

        self.time_encoder = TimeEncode(dimension=memory_dimension if use_memory else self.n_node_features).to(device)
        time_dim = memory_dimension if use_memory else self.n_node_features

        self.memory = None
        if use_memory:
            self.memory_dimension = memory_dimension
            raw_message_dim = 2 * memory_dimension + self.n_edge_features + time_dim
            msg_dim = message_dimension if message_function_type != "identity" else raw_message_dim
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=memory_dimension,
                device=device,
            )
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type, device=device)
            self.message_function = get_message_function(
                module_type=message_function_type,
                raw_message_dimension=raw_message_dim,
                message_dimension=msg_dim,
            )
            self.memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=msg_dim,
                memory_dimension=memory_dimension,
                device=device,
            )

        self.embedding_module = get_embedding_module(
            module_type=embedding_module_type,
            node_features=node_features,
            edge_features=edge_features,
            memory=self.memory,
            neighbor_finder=neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=time_dim,
            embedding_dimension=self.embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
            n_neighbors=self.n_neighbors,
        )
        self.affinity_score = MergeLayer(
            self.n_node_features, self.n_node_features, self.n_node_features, 1
        )

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    def update_memory(self, nodes: List[int], messages: dict):
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) == 0:
            return
        unique_messages = self.message_function.compute_message(unique_messages)
        self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)

    def get_updated_memory(self, nodes: List[int], messages: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)
        full_memory = self.memory.get_memory(list(range(self.n_nodes))).clone()
        full_ts = self._get_last_update_tensor().clone()
        if len(unique_nodes) == 0:
            return full_memory, full_ts
        unique_messages = self.message_function.compute_message(unique_messages)
        filtered_nodes, updated_memory, updated_ts = self.memory_updater.get_updated_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
        if len(filtered_nodes) == 0:
            return full_memory, full_ts
        idx = torch.tensor(filtered_nodes, device=self.device, dtype=torch.long)
        full_memory[idx] = updated_memory
        full_ts[idx] = updated_ts.squeeze()
        return full_memory, full_ts

    def _get_last_update_tensor(self) -> torch.Tensor:
        """Tensor [n_nodes] cho last_update (0 nếu chưa update)."""
        out = torch.zeros(self.n_nodes, device=self.device, dtype=torch.float32)
        for nid, t in self.memory._last_update.items():
            out[nid] = t.item() if hasattr(t, "item") else float(t)
        return out

    def get_raw_messages(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        edge_times_t = torch.from_numpy(edge_times).float().to(self.device)
        edge_feats = self.edge_raw_features[edge_idxs]
        source_memory = self.memory.get_memory(source_nodes)
        dest_memory = self.memory.get_memory(destination_nodes)
        source_time_delta = edge_times_t - self.memory.get_last_update(source_nodes)
        source_time_enc = self.time_encoder(source_time_delta.unsqueeze(1)).squeeze(1)
        source_message = torch.cat([source_memory, dest_memory, edge_feats, source_time_enc], dim=1)
        messages = defaultdict(list)
        for i in range(len(source_nodes)):
            messages[int(source_nodes[i])].append((source_message[i], edge_times_t[i]))
        return np.unique(source_nodes), messages

    def compute_temporal_embeddings(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        if self.use_memory:
            if self.memory_update_at_start:
                memory, _ = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
                self.update_memory(positives, self.memory.messages)
                self.memory.clear_messages(positives)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))

        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
        )

        source_emb = node_embedding[:n_samples]
        dest_emb = node_embedding[n_samples : 2 * n_samples]
        neg_emb = node_embedding[2 * n_samples :]

        if self.use_memory:
            unique_src, src_msgs = self.get_raw_messages(source_nodes, destination_nodes, edge_times, edge_idxs)
            unique_dst, dst_msgs = self.get_raw_messages(destination_nodes, source_nodes, edge_times, edge_idxs)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_src, src_msgs)
                self.memory.store_raw_messages(unique_dst, dst_msgs)
            else:
                self.update_memory(unique_src, src_msgs)
                self.update_memory(unique_dst, dst_msgs)

        return source_emb, dest_emb, neg_emb

    def compute_edge_probabilities(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_emb, dest_emb, neg_emb = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors
        )
        n_samples = len(source_nodes)
        pos_score = self.affinity_score(src_emb, dest_emb).squeeze(-1)
        neg_score = self.affinity_score(src_emb, neg_emb).squeeze(-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

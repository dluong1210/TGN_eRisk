"""
TGNUserSequence: TGN for eRisk user-level classification.

1 chuỗi conversations = 1 TGN. Memory target_user giữ nguyên qua các conversation.
Sau mỗi conversation, xóa memory của other users (tối ưu bộ nhớ).
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Optional, Tuple, Union

try:
    from ..modules.memory import Memory
    from ..modules.message_function import get_message_function
    from ..modules.message_aggregator import get_message_aggregator
    from ..modules.memory_updater import get_memory_updater
    from ..modules.embedding_module import TimeEncode, get_embedding_module
    from ..utils.utils import ClassificationHead
    from ..utils.neighbor_finder import get_temporal_ego_subgraph, get_neighbor_finder, NeighborFinder
except ImportError:
    from modules.memory import Memory
    from modules.message_function import get_message_function
    from modules.message_aggregator import get_message_aggregator
    from modules.memory_updater import get_memory_updater
    from modules.embedding_module import TimeEncode, get_embedding_module
    from utils.utils import ClassificationHead
    from utils.neighbor_finder import get_temporal_ego_subgraph, get_neighbor_finder, NeighborFinder


class TGNUserSequence(nn.Module):
    """
    TGN for user sequence: 1 UserData = 1 TGN.
    Memory target_user persists; other users freed after each conversation.
    """

    def __init__(
        self,
        n_users: int,
        edge_features: np.ndarray,
        device: torch.device,
        memory_dimension: int = 172,
        message_function_type: str = "identity",
        aggregator_type: str = "last",
        memory_updater_type: str = "gru",
        n_ego_layers: int = 2,
        embedding_module_type: str = "graph_attention",
        n_heads: int = 2,
        n_neighbors: int = 10,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users = n_users
        self.device = device
        self.n_ego_layers = n_ego_layers
        self.embedding_module_type = embedding_module_type
        self.n_neighbors = max(1, n_neighbors)

        self.edge_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.n_edge_features = self.edge_features.shape[1]

        self.memory_dimension = memory_dimension
        self.time_encoder = TimeEncode(dimension=memory_dimension).to(device)

        raw_message_dimension = 2 * memory_dimension + self.n_edge_features + memory_dimension
        if message_function_type == "identity":
            message_dimension = raw_message_dimension
        else:
            message_dimension = raw_message_dimension // 2

        self.memory = Memory(
            n_nodes=n_users,
            memory_dimension=memory_dimension,
            device=device,
        )
        self.message_function = get_message_function(
            module_type=message_function_type,
            raw_message_dimension=raw_message_dimension,
            message_dimension=message_dimension,
        )
        self.message_aggregator = get_message_aggregator(
            aggregator_type=aggregator_type,
            device=device,
        )
        self.memory_updater = get_memory_updater(
            module_type=memory_updater_type,
            memory=self.memory,
            message_dimension=message_dimension,
            memory_dimension=memory_dimension,
            device=device,
        )

        # Embedding module (TGN paper: identity / time / graph_attention / graph_sum)
        self.embedding_module = None
        if embedding_module_type in ("graph_attention", "graph_sum"):
            node_features = torch.zeros(n_users, memory_dimension, device=device, dtype=torch.float32)
            dummy_finder = NeighborFinder({}, uniform=False, n_nodes=n_users)
            self.embedding_module = get_embedding_module(
                module_type=embedding_module_type,
                node_features=node_features,
                edge_features=self.edge_features,
                memory=None,
                neighbor_finder=dummy_finder,
                time_encoder=self.time_encoder,
                n_layers=n_ego_layers,
                n_node_features=memory_dimension,
                n_edge_features=self.n_edge_features,
                n_time_features=memory_dimension,
                embedding_dimension=memory_dimension,
                device=device,
                n_heads=n_heads,
                dropout=dropout,
                use_memory=True,
            )

        self.classifier = ClassificationHead(
            input_dim=memory_dimension,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=dropout,
        )

    def reset_state(self):
        """Reset memory for new user/window (tạo buffer mới, xóa messages/last_update)."""
        self.memory.reset_state()

    def _get_raw_messages(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
    ) -> Tuple[List[int], dict, List[int], dict]:
        """Create raw messages for memory update."""
        edge_times_tensor = torch.from_numpy(edge_times).float().to(self.device)
        edge_feats = self.edge_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes)
        dest_memory = self.memory.get_memory(destination_nodes)

        source_time_delta = edge_times_tensor - self.memory.get_last_update(source_nodes)
        source_time_enc = self.time_encoder(source_time_delta).squeeze(1)

        dest_time_delta = edge_times_tensor - self.memory.get_last_update(destination_nodes)
        dest_time_enc = self.time_encoder(dest_time_delta).squeeze(1)

        source_message = torch.cat([
            source_memory, dest_memory, edge_feats, source_time_enc
        ], dim=1)
        dest_message = torch.cat([
            dest_memory, source_memory, edge_feats, dest_time_enc
        ], dim=1)

        source_messages = defaultdict(list)
        dest_messages = defaultdict(list)
        for i in range(len(source_nodes)):
            source_messages[int(source_nodes[i])].append((source_message[i], edge_times_tensor[i]))
            dest_messages[int(destination_nodes[i])].append((dest_message[i], edge_times_tensor[i]))

        return (
            np.unique(source_nodes).tolist(),
            source_messages,
            np.unique(destination_nodes).tolist(),
            dest_messages,
        )

    def _update_memory(self, nodes: List[int], messages: dict):
        """Update memory in-place."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) == 0:
            return
        unique_messages = self.message_function.compute_message(unique_messages)
        self.memory_updater.update_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )

    def _get_target_embedding(
        self,
        target_user: int,
        conv_context: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = None,
    ) -> torch.Tensor:
        """
        Get target user embedding.
        - identity: return memory directly (paper z_i = s_i).
        - graph_attention / graph_sum: use temporal graph aggregation (paper Eq 5-9).
        conv_context = (sources, dests, post_ids, timestamps, up_to_event_idx) for graph modules.
        """
        if self.embedding_module is None or conv_context is None:
            return self.memory.get_memory([target_user]).squeeze(0)

        sources, dests, post_ids, timestamps, up_to_event_idx = conv_context
        k = up_to_event_idx + 1
        if k <= 0:
            return self.memory.get_memory([target_user]).squeeze(0)

        # Build neighbor finder from current conv prefix (events 0..up_to_event_idx)
        cur_src = sources[:k]
        cur_dst = dests[:k]
        cur_eid = post_ids[:k]
        cur_ts = timestamps[:k]
        finder = get_neighbor_finder(cur_src, cur_dst, cur_eid, cur_ts, self.n_users, uniform=False)
        self.embedding_module.neighbor_finder = finder

        full_memory = self.memory.get_full_memory_tensor()
        current_ts = float(timestamps[up_to_event_idx])
        source_nodes = np.array([target_user], dtype=np.int64)
        timestamps_np = np.array([current_ts], dtype=np.float64)

        emb = self.embedding_module.compute_embedding(
            full_memory,
            source_nodes,
            timestamps_np,
            n_layers=self.n_ego_layers,
            n_neighbors=self.n_neighbors,
        )
        return emb.squeeze(0)

    def forward(
        self,
        user_data,
        return_per_event: bool = False,
        return_logits: bool = False,
    ) -> Union[List[torch.Tensor], List[Tuple[float, str, torch.Tensor]], torch.Tensor, None]:
        """
        Forward pass.

        Args:
            user_data: UserData with target_user and conversations
            return_per_event: If True (test), return (timestamp, conv_id, embedding) after each event.
            return_logits: If True (train/eval), chạy classifier trong forward và trả về logits [1, num_classes].
            window_aggregation: "last" hoặc "mean", dùng khi return_logits=True để gộp embeddings trước classifier.

        Returns:
            return_per_event=True: [(t1, conv_id1, emb1), ...]
            return_logits=False (mặc định): [emb_1, emb_2, ..., emb_K]
            return_logits=True: logits tensor [1, num_classes], hoặc None nếu không có embedding.
        """
        target_user = user_data.user_id
        conversations = user_data.get_conversations_sorted()

        if return_per_event:
            result: List[Tuple[float, str, torch.Tensor]] = []
        else:
            result = []

        for conv in conversations:
            if conv.n_interactions == 0:
                if not return_per_event:
                    emb = self._get_target_embedding(target_user)
                    result.append(emb)
                self.memory.free_nodes_except(target_user)
                continue

            sources, dests, post_ids, timestamps = get_temporal_ego_subgraph(
                conv.source_users,
                conv.dest_users,
                conv.post_ids,
                conv.timestamps,
                target_user,
                conv.end_time,
                self.n_ego_layers,
            )

            if len(sources) == 0:
                if not return_per_event:
                    emb = self._get_target_embedding(target_user)
                    result.append(emb)
                self.memory.free_nodes_except(target_user)
                continue

            conv_id_str = str(conv.conversation_id)
            conv_ctx = (sources, dests, post_ids, timestamps)

            for i in range(len(sources)):
                # Dùng slice thay vì np.array([...]) để giảm allocation trong vòng lặp
                batch_sources = sources[i : i + 1]
                batch_dests = dests[i : i + 1]
                batch_times = timestamps[i : i + 1].astype(np.float64)
                batch_post_ids = post_ids[i : i + 1]

                unique_src, src_msgs, unique_dst, dst_msgs = self._get_raw_messages(
                    batch_sources, batch_dests, batch_times, batch_post_ids
                )
                self.memory.store_raw_messages(unique_src, src_msgs)
                self.memory.store_raw_messages(unique_dst, dst_msgs)

                nodes_with_msgs = list(self.memory.messages.keys())
                self._update_memory(nodes_with_msgs, self.memory.messages)
                self.memory.clear_messages(nodes_with_msgs)

                if return_per_event:
                    emb = self._get_target_embedding(target_user, conv_context=(*conv_ctx, i))
                    result.append((float(timestamps[i]), conv_id_str, emb))

            if not return_per_event:
                emb = self._get_target_embedding(
                    target_user, conv_context=(*conv_ctx, len(sources) - 1)
                )
                result.append(emb)

            self.memory.free_nodes_except(target_user)

        if return_logits:
            if len(result) == 0:
                return None
            logits = self.classifier(result[-1].unsqueeze(0))
            return logits
        return result

"""
TGN Sequential Model for Depression Detection.

Hai approaches để xử lý chuỗi conversations:

1. CARRY-OVER: Embedding của target user sau mỗi conversation
   được dùng làm node features khởi tạo cho conversation sau.
   → Cuối cùng chỉ có 1 embedding duy nhất.

2. LSTM: Sau mỗi conversation thu được 1 embedding của target user.
   Chuỗi embeddings được feed qua LSTM, lấy hidden cuối để classify.
   → Capture sequential patterns across conversations.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from ..modules.memory import Memory
    from ..modules.message_function import get_message_function
    from ..modules.message_aggregator import get_message_aggregator
    from ..modules.memory_updater import get_memory_updater
    from ..modules.embedding_module import TimeEncode, get_embedding_module
    from ..utils.utils import ClassificationHead
    from ..utils.neighbor_finder import get_neighbor_finder, get_temporal_ego_subgraph
except ImportError:
    from modules.memory import Memory
    from modules.message_function import get_message_function
    from modules.message_aggregator import get_message_aggregator
    from modules.memory_updater import get_memory_updater
    from modules.embedding_module import TimeEncode, get_embedding_module
    from utils.utils import ClassificationHead
    from utils.neighbor_finder import get_neighbor_finder, get_temporal_ego_subgraph


class _MemoryView:
    """
    View chỉ các node đã update (ego). Index bằng node_id → trả về hàng tương ứng hoặc 0.
    Không cần full buffer (n_users, dim).
    """
    def __init__(self, device: torch.device, dim: int, updated_rows: Dict[int, torch.Tensor]):
        self._device = device
        self._dim = dim
        self.updated = updated_rows

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            indices, slice_obj = key
        else:
            indices = key
            slice_obj = slice(None)
        indices = np.asarray(indices).flatten()
        if indices.size == 0:
            return torch.empty(0, self._dim, device=self._device, dtype=torch.float32)
        out = torch.zeros(len(indices), self._dim, device=self._device, dtype=torch.float32)
        for i in range(len(indices)):
            nid = int(indices[i])
            if nid in self.updated:
                out[i] = self.updated[nid]
        if slice_obj != slice(None):
            return out[slice_obj]
        return out


class _NodeFeaturesView:
    """Chỉ lưu hàng cho node đã set (vd. target_user trong carryover), còn lại = 0."""
    def __init__(self, device: torch.device, dim: int, custom: Dict[int, torch.Tensor]):
        self._device = device
        self._dim = dim
        self.custom = custom

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            indices, slice_obj = key
        else:
            indices, slice_obj = key, slice(None)
        if hasattr(indices, 'cpu'):
            indices = indices.cpu().numpy()
        indices = np.asarray(indices).flatten()
        if indices.size == 0:
            return torch.empty(0, self._dim, device=self._device, dtype=torch.float32)
        out = torch.zeros(len(indices), self._dim, device=self._device, dtype=torch.float32)
        for i in range(len(indices)):
            nid = int(indices[i])
            if nid in self.custom:
                out[i] = self.custom[nid]
        if slice_obj != slice(None):
            return out[slice_obj]
        return out


class TGNSequential(nn.Module):
    """
    TGN with Sequential Conversation Processing.
    
    Supports two modes:
    1. 'carryover': Carry target user's embedding to next conversation
    2. 'lstm': Collect embeddings from each conversation, process with LSTM
    
    Tối ưu: Chỉ cần embedding của target_user nên khi n_layers in (0,1,2) chỉ tính
    L-hop ego (1-hop khi n_layers=1; 1+2-hop khi n_layers=2). Không cập nhật memory
    hay build neighbor_finder cho toàn bộ graph — chỉ edges/nodes trong ego.
    """
    
    def __init__(self,
                 n_users: int,
                 edge_features: np.ndarray,
                 device: torch.device,
                 # Sequence processing mode
                 sequence_mode: str = "carryover",  # 'carryover' or 'lstm'
                 # LSTM parameters (for lstm mode)
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 # TGN parameters
                 n_layers: int = 1,
                 n_heads: int = 2,
                 dropout: float = 0.1,
                 use_memory: bool = True,
                 memory_dimension: int = 172,
                 message_dimension: int = 100,
                 embedding_module_type: str = "graph_attention",
                     message_function_type: str = "identity",
                     aggregator_type: str = "last",
                     memory_updater_type: str = "gru",
                     n_neighbors: int = 10,
                     num_classes: int = 2,
                     conversation_batch_size: int = 200):
        """
        Args:
            sequence_mode: 'carryover' hoặc 'lstm'
                - carryover: Embedding target user → node features cho conv tiếp theo
                - lstm: Thu thập embeddings từ mỗi conv → LSTM → classify
            lstm_hidden_dim: Hidden dimension của LSTM (chỉ dùng cho lstm mode)
            lstm_num_layers: Số layers của LSTM
            lstm_bidirectional: Bidirectional LSTM hay không
            ... các tham số TGN khác
        """
        super(TGNSequential, self).__init__()
        
        self.n_users = n_users
        self.device = device
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.use_memory = use_memory
        self.sequence_mode = sequence_mode
        self.logger = logging.getLogger(__name__)
        self.conversation_batch_size = conversation_batch_size
        
        assert sequence_mode in ['carryover', 'lstm'], \
            f"sequence_mode must be 'carryover' or 'lstm', got {sequence_mode}"
        
        # Edge features (post embeddings)
        self.edge_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.n_edge_features = self.edge_features.shape[1]
        
        # Memory dimension
        if use_memory:
            self.memory_dimension = memory_dimension
            self.n_node_features = memory_dimension
        else:
            self.memory_dimension = 0
            self.n_node_features = self.n_edge_features
        
        # Node features — chỉ lưu cho node đã set (target_user trong carryover), còn lại = 0
        self._node_features_custom: Dict[int, torch.Tensor] = {}
        self.node_features = _NodeFeaturesView(device, self.n_node_features, self._node_features_custom)
        
        # Embedding dimension
        self.embedding_dimension = self.n_node_features
        
        # Time encoder
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.time_encoder = self.time_encoder.to(device)
        
        # Memory module
        self.memory = None
        self.message_aggregator = None
        self.message_function = None
        self.memory_updater = None
        
        if use_memory:
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.n_node_features
            
            if message_function_type == "identity":
                message_dimension = raw_message_dimension
            
            self.memory = Memory(
                n_nodes=n_users,
                memory_dimension=self.memory_dimension,
                device=device
            )
            
            self.message_aggregator = get_message_aggregator(
                aggregator_type=aggregator_type,
                device=device
            )
            
            self.message_function = get_message_function(
                module_type=message_function_type,
                raw_message_dimension=raw_message_dimension,
                message_dimension=message_dimension
            )
            
            self.memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=message_dimension,
                memory_dimension=self.memory_dimension,
                device=device
            )
        
        # Neighbor finder placeholder
        self.neighbor_finder = None
        
        # Embedding module params
        self.embedding_module_type = embedding_module_type
        self.embedding_module = None
        self._embedding_module_params = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'dropout': dropout,
            'use_memory': use_memory
        }
        
        # ========== LSTM for sequence mode ==========
        if sequence_mode == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.n_node_features,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=lstm_bidirectional,
                dropout=dropout if lstm_num_layers > 1 else 0
            )
            
            # LSTM output dimension
            lstm_output_dim = lstm_hidden_dim * (2 if lstm_bidirectional else 1)
            
            # Classification head for LSTM output
            self.classifier = ClassificationHead(
                input_dim=lstm_output_dim,
                hidden_dim=128,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            # Classification head for carryover mode
            self.classifier = ClassificationHead(
                input_dim=self.n_node_features,
                hidden_dim=128,
                num_classes=num_classes,
                dropout=dropout
            )
    
    def _init_embedding_module(self):
        """Initialize embedding module with current neighbor finder."""
        self.embedding_module = get_embedding_module(
            module_type=self.embedding_module_type,
            node_features=self.node_features,
            edge_features=self.edge_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=self._embedding_module_params['n_layers'],
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_node_features,
            embedding_dimension=self.embedding_dimension,
            device=self.device,
            n_heads=self._embedding_module_params['n_heads'],
            dropout=self._embedding_module_params['dropout'],
            use_memory=self._embedding_module_params['use_memory']
        )
        # Embedding module is created lazily during forward; ensure it's on same device
        self.embedding_module.to(self.device)
    
    def reset_state(self):
        """
        Reset all states for new user.
        Called at start of processing each target user.
        """
        # Reset memory
        if self.use_memory:
            self.memory.__init_memory__()
        
        # Reset node features (chỉ lưu ego/target_user nên clear dict)
        self._node_features_custom.clear()
    
    def get_raw_messages(self,
                         source_nodes: np.ndarray,
                         destination_nodes: np.ndarray,
                         edge_times: np.ndarray,
                         edge_idxs: np.ndarray) -> Tuple[List[int], Dict, List[int], Dict]:
        """Create raw messages for memory update."""
        edge_times_tensor = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_features[edge_idxs]
        
        source_memory = self.memory.get_memory(source_nodes)
        dest_memory = self.memory.get_memory(destination_nodes)
        
        source_time_delta = edge_times_tensor - self.memory.get_last_update(source_nodes)
        source_time_encoding = self.time_encoder(source_time_delta).squeeze(1)
        
        dest_time_delta = edge_times_tensor - self.memory.get_last_update(destination_nodes)
        dest_time_encoding = self.time_encoder(dest_time_delta).squeeze(1)
        
        source_message = torch.cat([
            source_memory, dest_memory, edge_features, source_time_encoding
        ], dim=1)
        
        dest_message = torch.cat([
            dest_memory, source_memory, edge_features, dest_time_encoding
        ], dim=1)
        
        source_messages = defaultdict(list)
        dest_messages = defaultdict(list)
        
        for i in range(len(source_nodes)):
            source_messages[int(source_nodes[i])].append(
                (source_message[i], edge_times_tensor[i])
            )
            dest_messages[int(destination_nodes[i])].append(
                (dest_message[i], edge_times_tensor[i])
            )
        # np.unique nhanh hơn set cho array lớn, thứ tự ổn định
        return (np.unique(source_nodes).tolist(), source_messages,
                np.unique(destination_nodes).tolist(), dest_messages)
    
    def get_updated_memory(self, nodes: List[int], messages: Dict) -> Tuple[_MemoryView, Optional[torch.Tensor]]:
        """Chỉ nodes có message (ego). Trả về view, không clone full buffer."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) == 0:
            return _MemoryView(self.device, self.memory.memory_dimension, {}), None
        unique_messages = self.message_function.compute_message(unique_messages)
        updated_rows, _ = self.memory_updater.get_updated_memory_rows_only(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
        if updated_rows.dtype != torch.float32:
            updated_rows = updated_rows.to(torch.float32)
        updated_dict = {int(nid): updated_rows[i] for i, nid in enumerate(unique_nodes)}
        return _MemoryView(self.device, self.memory.memory_dimension, updated_dict), None
    
    def update_memory(self, nodes: List[int], messages: Dict):
        """Update memory in-place."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        
        self.memory_updater.update_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
    
    def process_conversation(self,
                             conversation,
                             n_neighbors: int = None,
                             target_user: Optional[int] = None):
        """
        Process a single conversation (neighbor_finder + memory), trả về end_time.
        
        Tối ưu: Nếu target_user và n_layers in (0,1,2) thì chỉ xử lý L-hop ego subgraph.
        
        Args:
            conversation: Conversation object
            n_neighbors: Number of neighbors to sample
            target_user: Nếu có, dùng để lọc ego subgraph khi n_layers in (0,1,2)
        
        Returns:
            conversation.end_time (để gọi compute_user_embedding(target_user, end_time) sau)
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        sources = conversation.source_users
        dests = conversation.dest_users
        timestamps = conversation.timestamps
        post_ids = conversation.post_ids
        
        if len(sources) == 0:
            return None
        
        end_time = conversation.end_time
        sources_use, dests_use, edge_idxs_use, timestamps_use = sources, dests, post_ids, timestamps
        # Chỉ dùng L-hop ego (1-hop khi n_layers=1, 1+2-hop khi n_layers=2) — không tính toàn graph
        if target_user is not None and 0 <= self.n_layers <= 2:
            sources_use, dests_use, edge_idxs_use, timestamps_use = get_temporal_ego_subgraph(
                sources, dests, post_ids, timestamps, target_user, end_time, self.n_layers
            )
        
        self.neighbor_finder = get_neighbor_finder(
            sources=sources_use,
            destinations=dests_use,
            edge_idxs=edge_idxs_use,
            timestamps=timestamps_use,
            n_nodes=self.n_users,
            uniform=False
        )
        self._init_embedding_module()
        
        batch_size = self.conversation_batch_size
        n_interactions = len(sources_use)
        if n_interactions <= batch_size:
            batch_size = n_interactions
        
        for start_idx in range(0, n_interactions, batch_size):
            end_idx = min(start_idx + batch_size, n_interactions)
            batch_sources = sources_use[start_idx:end_idx]
            batch_dests = dests_use[start_idx:end_idx]
            batch_timestamps = timestamps_use[start_idx:end_idx]
            batch_post_ids = edge_idxs_use[start_idx:end_idx]
            if self.use_memory:
                positives = np.concatenate([batch_sources, batch_dests])
                unique_positives = np.unique(positives)
                self.update_memory(unique_positives.tolist(), self.memory.messages)
                self.memory.clear_messages(unique_positives.tolist())
                unique_sources, source_msgs, unique_dests, dest_msgs = self.get_raw_messages(
                    batch_sources, batch_dests, batch_timestamps, batch_post_ids
                )
                self.memory.store_raw_messages(unique_sources, source_msgs)
                self.memory.store_raw_messages(unique_dests, dest_msgs)
        
        return end_time
    
    def process_conversations_batch(self,
                                    conversations: List,
                                    target_user: int,
                                    n_neighbors: int = None,
                                    use_larger_batches: bool = True) -> List[torch.Tensor]:
        """
        Process multiple conversations efficiently (for LSTM mode).
        
        Tối ưu hóa:
        - Xử lý interactions trong batch lớn hơn (tận dụng GPU tốt hơn với conversation_batch_size)
        - Reset memory giữa các conversations để đảm bảo độc lập
        - Với use_larger_batches=True: tăng batch_size động dựa trên số lượng conversations
        
        QUAN TRỌNG: Mỗi conversation phải có neighbor_finder riêng để đảm bảo độc lập.
        Không thể gom tất cả conversations lại vì neighbor_finder sẽ chứa edges từ
        conversations khác, vi phạm tính độc lập của LSTM mode.
        
        Args:
            conversations: List of Conversation objects (đã sorted theo thời gian)
            target_user: Target user ID
            n_neighbors: Number of neighbors to sample
            use_larger_batches: Nếu True, tăng batch_size động để xử lý nhiều interactions hơn
        
        Returns:
            List of embeddings [embedding_1, embedding_2, ...] tại end_time của mỗi conversation
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        if len(conversations) == 0:
            return []
        
        # Filter out empty conversations
        valid_convs = [c for c in conversations if c.n_interactions > 0]
        if len(valid_convs) == 0:
            return []
        
        # Tối ưu: tăng batch_size động nếu có nhiều conversations nhỏ
        base_batch_size = self.conversation_batch_size
        if use_larger_batches and len(valid_convs) > 1:
            # Tính tổng interactions và điều chỉnh batch_size
            total_interactions = sum(len(c.source_users) for c in valid_convs)
            avg_interactions_per_conv = total_interactions / len(valid_convs)
            # Nếu conversations nhỏ, tăng batch_size để xử lý nhiều hơn
            if avg_interactions_per_conv < base_batch_size:
                # Tăng batch_size lên 2-4x nếu conversations nhỏ
                dynamic_batch_size = min(base_batch_size * 2, max(base_batch_size, total_interactions // len(valid_convs) + 100))
            else:
                dynamic_batch_size = base_batch_size
        else:
            dynamic_batch_size = base_batch_size
        
        # Xử lý từng conversation tuần tự (mỗi conversation độc lập về memory và neighbor_finder)
        conversation_embeddings = []
        
        for conv in valid_convs:
            # Reset memory trước mỗi conversation (đảm bảo độc lập)
            if self.use_memory:
                self.memory.__init_memory__()
            
            # Tối ưu: chỉ dùng L-hop ego subgraph (layer 0,1,2) để tránh tính toán thừa
            end_time = conv.end_time
            sources_use = conv.source_users
            dests_use = conv.dest_users
            edge_idxs_use = conv.post_ids
            timestamps_use = conv.timestamps
            # Chỉ dùng L-hop ego — không build graph toàn bộ conversation
            if 0 <= self.n_layers <= 2:
                sources_use, dests_use, edge_idxs_use, timestamps_use = get_temporal_ego_subgraph(
                    conv.source_users,
                    conv.dest_users,
                    conv.post_ids,
                    conv.timestamps,
                    target_user,
                    end_time,
                    self.n_layers,
                )
            
            # NeighborFinder chỉ chứa edges trong ego
            self.neighbor_finder = get_neighbor_finder(
                sources=sources_use,
                destinations=dests_use,
                edge_idxs=edge_idxs_use,
                timestamps=timestamps_use,
                n_nodes=self.n_users,
                uniform=False
            )
            self._init_embedding_module()
            
            # Process interactions (chỉ ego subgraph khi n_layers in (0,1,2))
            batch_size = dynamic_batch_size
            n_conv_interactions = len(sources_use)
            
            for start_idx in range(0, n_conv_interactions, batch_size):
                end_idx = min(start_idx + batch_size, n_conv_interactions)
                
                batch_sources = sources_use[start_idx:end_idx]
                batch_dests = dests_use[start_idx:end_idx]
                batch_timestamps = timestamps_use[start_idx:end_idx]
                batch_post_ids = edge_idxs_use[start_idx:end_idx]
                
                if self.use_memory:
                    positives = np.concatenate([batch_sources, batch_dests])
                    unique_positives = np.unique(positives)
                    self.update_memory(unique_positives.tolist(), self.memory.messages)
                    self.memory.clear_messages(unique_positives.tolist())
                    
                    unique_sources, source_msgs, unique_dests, dest_msgs = self.get_raw_messages(
                        batch_sources, batch_dests, batch_timestamps, batch_post_ids
                    )
                    
                    self.memory.store_raw_messages(unique_sources, source_msgs)
                    self.memory.store_raw_messages(unique_dests, dest_msgs)
            
            # Lấy embedding tại end_time của conversation này
            user_embedding = self.compute_user_embedding(
                target_user, end_time, n_neighbors
            )
            conversation_embeddings.append(user_embedding)
        
        return conversation_embeddings
    
    def compute_user_embedding(self,
                               user_id: int,
                               timestamp: float,
                               n_neighbors: int = None) -> torch.Tensor:
        """
        Compute embedding for a user at given timestamp.
        Chỉ cần memory cho nodes trong L-hop ego (có pending messages) — không duyệt toàn bộ n_users.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        user_ids = np.array([user_id])
        timestamps = np.array([timestamp])
        
        if self.use_memory:
            # Chỉ aggregate/update memory cho nodes có pending messages (chỉ node trong ego)
            nodes_with_messages = list(self.memory.messages.keys()) if self.memory.messages else []
            memory, _ = self.get_updated_memory(nodes_with_messages, self.memory.messages)
        else:
            memory = None
        
        # Update embedding module với node_features hiện tại
        self.embedding_module.node_features = self.node_features
        
        embeddings = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=user_ids,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors
        )
        
        return embeddings
    
    def forward_carryover(self,
                          user_data,
                          n_neighbors: int = None) -> torch.Tensor:
        """
        Forward pass using CARRY-OVER mode.
        
        Sau mỗi conversation, embedding của target user được gán vào
        node_features để dùng làm khởi tạo cho conversation sau.
        
        Args:
            user_data: UserData object
            n_neighbors: Number of neighbors
        
        Returns:
            Logits [1, num_classes]
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        target_user = user_data.user_id
        conversations = user_data.get_conversations_sorted()
        
        if len(conversations) == 0 or user_data.total_interactions == 0:
            zero_emb = torch.zeros(1, self.n_node_features).to(self.device)
            return self.classifier(zero_emb)
        
        # Process each conversation sequentially
        last_embedding = None
        for conv in conversations:
            if conv.n_interactions == 0:
                continue
            
            # Process conversation (updates memory; ego subgraph khi n_layers in (0,1,2))
            end_time = self.process_conversation(conv, n_neighbors, target_user=target_user)
            
            # Compute target user embedding sau conversation này
            user_embedding = self.compute_user_embedding(
                target_user, end_time, n_neighbors
            )
            
            # ===== CARRY-OVER: Chỉ lưu node features cho target user =====
            self._node_features_custom[target_user] = user_embedding.squeeze(0).detach()
            
            last_embedding = user_embedding  # Giữ để classify từ tensor có grad
            
            # Reset memory cho conversation mới (giữ node_features đã update)
            if self.use_memory:
                self.memory.__init_memory__()
        
        # Final embedding: dùng embedding của conversation cuối (có grad) để classify
        if last_embedding is not None:
            final_embedding = last_embedding
        else:
            emb = self._node_features_custom.get(
                target_user, torch.zeros(self.n_node_features, device=self.device)
            )
            final_embedding = emb.unsqueeze(0)
        
        return self.classifier(final_embedding)
    
    def forward_lstm(self,
                     user_data,
                     n_neighbors: int = None,
                     use_batch_processing: bool = True) -> torch.Tensor:
        """
        Forward pass using LSTM mode.
        
        Thu thập embedding của target user sau mỗi conversation,
        tạo thành sequence, feed qua LSTM để classify.
        
        Args:
            user_data: UserData object
            n_neighbors: Number of neighbors
            use_batch_processing: Nếu True, dùng process_conversations_batch (tối ưu với batch interactions)
        
        Returns:
            Logits [1, num_classes]
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        target_user = user_data.user_id
        conversations = user_data.get_conversations_sorted()
        
        if len(conversations) == 0 or user_data.total_interactions == 0:
            zero_emb = torch.zeros(1, self.n_node_features).to(self.device)
            # Pass through LSTM with single zero embedding
            lstm_out, (h_n, c_n) = self.lstm(zero_emb.unsqueeze(0))
            # Get last hidden state
            if self.lstm.bidirectional:
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                final_hidden = h_n[-1]
            # Ensure correct shape [1, hidden_dim]
            if final_hidden.dim() == 1:
                final_hidden = final_hidden.unsqueeze(0)
            return self.classifier(final_hidden)
        
        # Collect embeddings from each conversation
        if use_batch_processing and len(conversations) > 1:
            # Xử lý conversations với process_conversations_batch (tối ưu với batch interactions)
            # Sử dụng larger batches để tối ưu hóa tốc độ
            conversation_embeddings = self.process_conversations_batch(
                conversations, target_user, n_neighbors, use_larger_batches=True
            )
        else:
            # Xử lý tuần tự từng conversation (backward compatibility)
            conversation_embeddings = []
            
            for conv in conversations:
                if conv.n_interactions == 0:
                    continue
                
                # Reset memory trước mỗi conversation (QUAN TRỌNG: đảm bảo độc lập)
                if self.use_memory:
                    self.memory.__init_memory__()
                
                # Process conversation (tạo neighbor_finder riêng; ego subgraph khi n_layers in (0,1,2))
                end_time = self.process_conversation(conv, n_neighbors, target_user=target_user)
                
                # Compute target user embedding
                user_embedding = self.compute_user_embedding(
                    target_user, end_time, n_neighbors
                )
                
                conversation_embeddings.append(user_embedding)
        
        if len(conversation_embeddings) == 0:
            zero_emb = torch.zeros(1, self.n_node_features).to(self.device)
            lstm_out, (h_n, c_n) = self.lstm(zero_emb.unsqueeze(0))
            if self.lstm.bidirectional:
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                final_hidden = h_n[-1]
            if final_hidden.dim() == 1:
                final_hidden = final_hidden.unsqueeze(0)
            return self.classifier(final_hidden)
        
        # Stack embeddings: [1, n_conversations, embedding_dim]
        embeddings_sequence = torch.cat(conversation_embeddings, dim=0).unsqueeze(0)
        
        # LSTM forward
        # lstm_out: [1, seq_len, hidden_dim * num_directions]
        # h_n: [num_layers * num_directions, 1, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(embeddings_sequence)
        
        # Get final hidden state
        if self.lstm.bidirectional:
            # Concatenate last hidden states from both directions
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_hidden = h_n[-1]
        
        # Ensure correct shape [1, hidden_dim * num_directions]
        if final_hidden.dim() == 1:
            final_hidden = final_hidden.unsqueeze(0)
        
        return self.classifier(final_hidden)
    
    def forward(self, user_data, n_neighbors: int = None) -> torch.Tensor:
        """
        Forward pass (used by DDP / DataParallel). Delegates to forward_user.
        """
        return self.forward_user(user_data, n_neighbors)

    def forward_batch_users(self,
                           users_data: List,
                           n_neighbors: int = None,
                           use_parallel_conversations: bool = False,
                           max_workers: int = 4) -> torch.Tensor:
        """
        Forward pass cho batch nhiều users (chỉ hỗ trợ LSTM mode).
        
        Xử lý tuần tự từng user nhưng gom kết quả để tính batch loss.
        Mỗi user được reset state riêng để đảm bảo độc lập.
        
        Tối ưu hóa:
        - Sử dụng batch processing cho conversations (tăng conversation_batch_size động)
        - use_parallel_conversations: Hiện tại chưa hỗ trợ true parallelism (do CUDA context),
          nhưng có thể tối ưu bằng cách tăng batch sizes
        
        Args:
            users_data: List of UserData objects
            n_neighbors: Number of neighbors
            use_parallel_conversations: (Reserved for future use) Parallel conversation processing
            max_workers: (Reserved for future use) Max workers for parallel processing
        
        Returns:
            Logits [batch_size, num_classes]
        """
        if self.sequence_mode != 'lstm':
            raise ValueError("forward_batch_users chỉ hỗ trợ LSTM mode")
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        batch_size = len(users_data)
        if batch_size == 0:
            return torch.zeros(0, 2).to(self.device)
        
        # Xử lý tuần tự (AN TOÀN và vẫn nhanh nhờ batch gradient)
        # Tối ưu: sử dụng batch processing với larger batches
        all_logits = []
        
        for user_data in users_data:
            # Reset state cho mỗi user (quan trọng!)
            self.reset_state()
            # Sử dụng batch processing với larger batches để tối ưu
            logits = self.forward_lstm(user_data, n_neighbors, use_batch_processing=True)
            all_logits.append(logits)
        
        # Stack: [batch_size, num_classes]
        return torch.cat(all_logits, dim=0)
    
    def forward_merged_graph(self,
                            batch_user_data: List,
                            n_neighbors: int = None) -> torch.Tensor:
        """
        Forward pass trên MỘT graph lớn gộp từ nhiều users: gộp mọi conversation trong batch
        thành một timeline events, chạy TGN một lần, snapshot embedding tại mỗi conversation end.
        Nhanh hơn rất nhiều so với chạy TGN từng conversation / từng user.
        
        Hỗ trợ cả carryover và lstm mode.
        
        Args:
            batch_user_data: List of UserData (cùng format như forward_batch_users)
            n_neighbors: Number of neighbors
        
        Returns:
            Logits [batch_size, num_classes]
        """
        try:
            from ..utils.data_structures import merge_user_data_to_graph
        except ImportError:
            from utils.data_structures import merge_user_data_to_graph
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        batch_size = len(batch_user_data)
        if batch_size == 0:
            return torch.zeros(0, 2).to(self.device)
        
        sources, destinations, timestamps, post_ids, conv_ends, user_ids, _ = \
            merge_user_data_to_graph(batch_user_data)
        
        if len(sources) == 0:
            zero_emb = torch.zeros(1, self.n_node_features).to(self.device)
            if self.sequence_mode == 'lstm':
                lstm_out, (h_n, c_n) = self.lstm(zero_emb.unsqueeze(0))
                final_hidden = h_n[-1] if not self.lstm.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=-1)
                if final_hidden.dim() == 1:
                    final_hidden = final_hidden.unsqueeze(0)
                logits = self.classifier(final_hidden)
            else:
                logits = self.classifier(zero_emb)
            return logits.expand(batch_size, -1)
        
        self.reset_state()
        self.neighbor_finder = get_neighbor_finder(
            sources=sources,
            destinations=destinations,
            edge_idxs=post_ids,
            timestamps=timestamps,
            n_nodes=self.n_users,
            uniform=False,
        )
        self._init_embedding_module()
        
        conv_ends_queue = list(conv_ends)
        embeddings_per_user = [dict() for _ in range(batch_size)]  # user_idx_in_batch -> {conv_idx: tensor}
        batch_sz = self.conversation_batch_size
        event_idx = 0
        n_events = len(sources)
        
        while event_idx < n_events:
            next_conv_time = conv_ends_queue[0][0] if conv_ends_queue else float('inf')
            j = event_idx
            # Khi có conv end: lấy hết events với t <= next_conv_time (để snapshot đúng thời điểm)
            # Khi không có: lấy tối đa batch_sz events
            if next_conv_time != float('inf'):
                while j < n_events and timestamps[j] <= next_conv_time:
                    j += 1
                if j == event_idx:
                    # Không còn event nào có t <= next_conv_time → snapshot rồi xử lý event tiếp theo
                    while conv_ends_queue and conv_ends_queue[0][0] <= next_conv_time:
                        end_time, user_idx_in_batch, conv_idx = conv_ends_queue.pop(0)
                        uid = user_ids[user_idx_in_batch]
                        emb = self.compute_user_embedding(uid, end_time, n_neighbors)
                        embeddings_per_user[user_idx_in_batch][conv_idx] = emb
                    j = min(event_idx + 1, n_events)
            else:
                j = min(event_idx + batch_sz, n_events)
            
            batch_sources = sources[event_idx:j]
            batch_dests = destinations[event_idx:j]
            batch_timestamps = timestamps[event_idx:j]
            batch_post_ids = post_ids[event_idx:j]
            
            if self.use_memory and len(batch_sources) > 0:
                positives = np.concatenate([batch_sources, batch_dests])
                unique_positives = np.unique(positives)
                self.update_memory(unique_positives.tolist(), self.memory.messages)
                self.memory.clear_messages(unique_positives.tolist())
                unique_sources, source_msgs, unique_dests, dest_msgs = self.get_raw_messages(
                    batch_sources, batch_dests, batch_timestamps, batch_post_ids
                )
                self.memory.store_raw_messages(unique_sources, source_msgs)
                self.memory.store_raw_messages(unique_dests, dest_msgs)
            
            max_time_batch = float(batch_timestamps[-1]) if len(batch_timestamps) > 0 else float('-inf')
            while conv_ends_queue and conv_ends_queue[0][0] <= max_time_batch:
                end_time, user_idx_in_batch, conv_idx = conv_ends_queue.pop(0)
                uid = user_ids[user_idx_in_batch]
                emb = self.compute_user_embedding(uid, end_time, n_neighbors)
                embeddings_per_user[user_idx_in_batch][conv_idx] = emb
            
            event_idx = j
        
        # Build logits per user từ embeddings_per_user
        all_logits = []
        zero_emb = torch.zeros(1, self.n_node_features).to(self.device)
        
        for user_idx_in_batch in range(batch_size):
            conv_to_emb = embeddings_per_user[user_idx_in_batch]
            if not conv_to_emb:
                if self.sequence_mode == 'lstm':
                    lstm_out, (h_n, c_n) = self.lstm(zero_emb.unsqueeze(0))
                    final_hidden = h_n[-1] if not self.lstm.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=-1)
                    if final_hidden.dim() == 1:
                        final_hidden = final_hidden.unsqueeze(0)
                    all_logits.append(self.classifier(final_hidden))
                else:
                    all_logits.append(self.classifier(zero_emb))
                continue
            
            sorted_conv_idxs = sorted(conv_to_emb.keys())
            conv_embeddings = [conv_to_emb[k] for k in sorted_conv_idxs]
            seq = torch.cat(conv_embeddings, dim=0).unsqueeze(0)  # [1, n_conv, dim]
            
            if self.sequence_mode == 'lstm':
                lstm_out, (h_n, c_n) = self.lstm(seq)
                final_hidden = h_n[-1] if not self.lstm.bidirectional else torch.cat([h_n[-2], h_n[-1]], dim=-1)
                if final_hidden.dim() == 1:
                    final_hidden = final_hidden.unsqueeze(0)
                all_logits.append(self.classifier(final_hidden))
            else:
                last_emb = seq[0, -1:, :]
                all_logits.append(self.classifier(last_emb))
        
        return torch.cat(all_logits, dim=0)
    
    def forward_user(self,
                     user_data,
                     n_neighbors: int = None) -> torch.Tensor:
        """
        Main forward pass - dispatches to appropriate mode.

        Args:
            user_data: UserData object
            n_neighbors: Number of neighbors

        Returns:
            Logits [1, num_classes]
        """
        if self.sequence_mode == 'carryover':
            return self.forward_carryover(user_data, n_neighbors)
        else:
            return self.forward_lstm(user_data, n_neighbors)
    
    def detach_memory(self):
        """Detach memory from computation graph."""
        if self.use_memory:
            self.memory.detach_memory()


class TGNCarryOver(TGNSequential):
    """Convenience class for Carry-Over mode."""
    
    def __init__(self, *args, **kwargs):
        kwargs['sequence_mode'] = 'carryover'
        super().__init__(*args, **kwargs)


class TGNLstm(TGNSequential):
    """Convenience class for LSTM mode."""
    
    def __init__(self, *args, **kwargs):
        kwargs['sequence_mode'] = 'lstm'
        super().__init__(*args, **kwargs)

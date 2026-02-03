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
    from ..utils.neighbor_finder import get_neighbor_finder
except ImportError:
    from modules.memory import Memory
    from modules.message_function import get_message_function
    from modules.message_aggregator import get_message_aggregator
    from modules.memory_updater import get_memory_updater
    from modules.embedding_module import TimeEncode, get_embedding_module
    from utils.utils import ClassificationHead
    from utils.neighbor_finder import get_neighbor_finder


class TGNSequential(nn.Module):
    """
    TGN with Sequential Conversation Processing.
    
    Supports two modes:
    1. 'carryover': Carry target user's embedding to next conversation
    2. 'lstm': Collect embeddings from each conversation, process with LSTM
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
        
        # Node features - có thể được cập nhật trong carryover mode
        # Dùng buffer thay vì parameter để có thể modify
        self.register_buffer(
            'node_features',
            torch.zeros((n_users, self.n_node_features))
        )
        
        # Lưu node features gốc để reset
        self.register_buffer(
            'node_features_initial',
            torch.zeros((n_users, self.n_node_features))
        )
        
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
        
        # Reset node features to initial values
        self.node_features.copy_(self.node_features_initial)
    
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
        
        source_time_delta = edge_times_tensor - self.memory.last_update[source_nodes]
        source_time_encoding = self.time_encoder(source_time_delta).squeeze(1)
        
        dest_time_delta = edge_times_tensor - self.memory.last_update[destination_nodes]
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
            source_messages[source_nodes[i]].append(
                (source_message[i], edge_times_tensor[i])
            )
            dest_messages[destination_nodes[i]].append(
                (dest_message[i], edge_times_tensor[i])
            )
        
        return (list(set(source_nodes)), source_messages,
                list(set(destination_nodes)), dest_messages)
    
    def get_updated_memory(self, nodes: List[int], messages: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get updated memory without persisting."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
        
        return updated_memory, updated_last_update
    
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
                             n_neighbors: int = None) -> torch.Tensor:
        """
        Process a single conversation and return target user embedding.
        
        Args:
            conversation: Conversation object
            n_neighbors: Number of neighbors to sample
        
        Returns:
            Embedding of target user after this conversation [1, embedding_dim]
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        # Get conversation data
        sources = conversation.source_users
        dests = conversation.dest_users
        timestamps = conversation.timestamps
        post_ids = conversation.post_ids
        
        if len(sources) == 0:
            return None
        
        # Create neighbor finder for this conversation
        self.neighbor_finder = get_neighbor_finder(
            sources=sources,
            destinations=dests,
            edge_idxs=post_ids,
            timestamps=timestamps,
            n_nodes=self.n_users,
            uniform=False
        )
        self._init_embedding_module()
        
        # Process interactions in batches
        batch_size = self.conversation_batch_size
        n_interactions = len(sources)
        
        for start_idx in range(0, n_interactions, batch_size):
            end_idx = min(start_idx + batch_size, n_interactions)
            
            batch_sources = sources[start_idx:end_idx]
            batch_dests = dests[start_idx:end_idx]
            batch_timestamps = timestamps[start_idx:end_idx]
            batch_post_ids = post_ids[start_idx:end_idx]
            
            if self.use_memory:
                positives = np.concatenate([batch_sources, batch_dests])
                self.update_memory(positives.tolist(), self.memory.messages)
                self.memory.clear_messages(positives.tolist())
                
                unique_sources, source_msgs, unique_dests, dest_msgs = self.get_raw_messages(
                    batch_sources, batch_dests, batch_timestamps, batch_post_ids
                )
                
                self.memory.store_raw_messages(unique_sources, source_msgs)
                self.memory.store_raw_messages(unique_dests, dest_msgs)
        
        # Return embedding at end of conversation
        return conversation.end_time
    
    def compute_user_embedding(self,
                               user_id: int,
                               timestamp: float,
                               n_neighbors: int = None) -> torch.Tensor:
        """Compute embedding for a user at given timestamp."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        user_ids = np.array([user_id])
        timestamps = np.array([timestamp])
        
        if self.use_memory:
            memory, _ = self.get_updated_memory(
                list(range(self.n_users)), self.memory.messages
            )
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
            
            # Process conversation (updates memory)
            end_time = self.process_conversation(conv, n_neighbors)
            
            # Compute target user embedding sau conversation này
            user_embedding = self.compute_user_embedding(
                target_user, end_time, n_neighbors
            )
            
            # ===== CARRY-OVER: Update node features cho target user =====
            # Embedding này sẽ được dùng làm khởi tạo cho conversation tiếp theo.
            # .detach() khi ghi buffer để tránh giữ computation graph qua nhiều conversation.
            self.node_features[target_user] = user_embedding.squeeze(0).detach()
            
            last_embedding = user_embedding  # Giữ để classify từ tensor có grad
            
            # Reset memory cho conversation mới (giữ node_features đã update)
            if self.use_memory:
                self.memory.__init_memory__()
        
        # Final embedding: dùng embedding của conversation cuối (có grad) để classify
        if last_embedding is not None:
            final_embedding = last_embedding
        else:
            final_embedding = self.node_features[target_user].unsqueeze(0)
        
        return self.classifier(final_embedding)
    
    def forward_lstm(self,
                     user_data,
                     n_neighbors: int = None) -> torch.Tensor:
        """
        Forward pass using LSTM mode.
        
        Thu thập embedding của target user sau mỗi conversation,
        tạo thành sequence, feed qua LSTM để classify.
        
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
        conversation_embeddings = []
        
        for conv in conversations:
            if conv.n_interactions == 0:
                continue
            
            # Process conversation
            end_time = self.process_conversation(conv, n_neighbors)
            
            # Compute target user embedding
            user_embedding = self.compute_user_embedding(
                target_user, end_time, n_neighbors
            )
            
            conversation_embeddings.append(user_embedding)
            
            # Reset memory for next conversation
            # (Trong LSTM mode, mỗi conv được xử lý độc lập về memory)
            if self.use_memory:
                self.memory.__init_memory__()
        
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

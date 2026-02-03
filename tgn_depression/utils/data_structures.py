"""
Data structures for TGN Depression Detection.

Target users là ĐỘC LẬP: mỗi target_user có chuỗi conversations riêng, trong đó
họ tương tác với các user khác trên social (không phải target_user tương tác
với nhau). Mỗi target user có MỘT label duy nhất (depression hay không).

Trong mỗi conversation của một target user:
- Nodes: target user + các user khác (trên mạng) tham gia conversation
- Edges: posts/replies giữa users (có timestamps)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


@dataclass
class Conversation:
    """
    Represents a single conversation (temporal graph).
    
    Trong mỗi conversation:
    - Nodes: users tham gia
    - Edges: posts/replies giữa users (có timestamps)
    """
    conversation_id: str
    
    # Interaction data (sorted by timestamp)
    source_users: np.ndarray      # User gửi post/reply
    dest_users: np.ndarray        # User nhận (được reply)
    timestamps: np.ndarray        # Timestamp của mỗi interaction
    post_ids: np.ndarray          # ID của post (để lấy embedding)
    
    # Metadata
    n_interactions: int = field(init=False)
    unique_users: set = field(init=False)
    start_time: float = field(init=False)
    end_time: float = field(init=False)
    
    def __post_init__(self):
        self.n_interactions = len(self.source_users)
        self.unique_users = set(self.source_users) | set(self.dest_users)
        self.start_time = self.timestamps.min() if len(self.timestamps) > 0 else 0.0
        self.end_time = self.timestamps.max() if len(self.timestamps) > 0 else 0.0


@dataclass
class UserData:
    """
    Represents all data for a single user (TARGET USER).
    
    - user_id: ID của user cần dự đoán
    - conversations: Danh sách các conversations user tham gia (sorted by time)
    - label: 0 (không depression) hoặc 1 (depression)
    """
    user_id: int                  # User index (đã map từ string)
    user_id_str: str              # User ID gốc (string)
    conversations: List[Conversation]
    label: int                    # Depression label (0 or 1)
    
    @property
    def n_conversations(self) -> int:
        return len(self.conversations)
    
    @property
    def total_interactions(self) -> int:
        return sum(c.n_interactions for c in self.conversations)
    
    def get_conversations_sorted(self) -> List[Conversation]:
        """
        Return conversations in the stored order.
        
        NOTE: We intentionally do NOT sort conversations by time here.
        The conversation sequence is assumed to follow the order in the input data.
        """
        return list(self.conversations)
    
    def get_evaluation_time(self) -> float:
        """Thời điểm cuối cùng để evaluate user."""
        if self.n_conversations == 0:
            return 0.0
        return max(c.end_time for c in self.conversations)


class UserBatch:
    """
    Batch of users for training.
    """
    def __init__(self, users: List[UserData]):
        self.users = users
        self.batch_size = len(users)
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        return self.users[idx]
    
    def get_labels(self) -> torch.Tensor:
        """Get all labels in batch."""
        return torch.LongTensor([u.label for u in self.users])


class DepressionDataset:
    """
    Dataset for Depression Detection using TGN.
    
    Structure:
    - Mỗi sample là MỘT USER với TẤT CẢ conversations của user đó
    - Label là depression status của user
    """
    
    def __init__(self, 
                 users: List[UserData],
                 post_embeddings: np.ndarray,
                 n_total_users: int,
                 user_to_idx: Dict[str, int],
                 idx_to_user: Dict[int, str]):
        """
        Args:
            users: List of UserData objects (target users)
            post_embeddings: numpy array [n_posts, embedding_dim] - Pre-computed embeddings
            n_total_users: Total number of users in entire dataset (including non-targets)
            user_to_idx: Mapping from user_id string to index
            idx_to_user: Reverse mapping
        """
        self.users = users
        self.post_embeddings = post_embeddings
        self.n_total_users = n_total_users
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        
        self.n_target_users = len(users)
        self.embedding_dim = post_embeddings.shape[1] if len(post_embeddings) > 0 else 0
        
        # Statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        self.n_depression = sum(1 for u in self.users if u.label == 1)
        self.n_non_depression = self.n_target_users - self.n_depression
        
        total_conversations = sum(u.n_conversations for u in self.users)
        self.avg_conversations_per_user = total_conversations / max(self.n_target_users, 1)
        
        total_interactions = sum(u.total_interactions for u in self.users)
        self.avg_interactions_per_user = total_interactions / max(self.n_target_users, 1)
    
    def __len__(self):
        return self.n_target_users
    
    def __getitem__(self, idx) -> UserData:
        return self.users[idx]
    
    def get_post_embedding(self, post_id: int) -> np.ndarray:
        """Get embedding for a specific post."""
        return self.post_embeddings[post_id]
    
    def get_edge_features(self, post_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for multiple posts (edge features)."""
        return self.post_embeddings[post_ids]
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return {
            'n_target_users': self.n_target_users,
            'n_total_users': self.n_total_users,
            'n_depression': self.n_depression,
            'n_non_depression': self.n_non_depression,
            'depression_ratio': self.n_depression / max(self.n_target_users, 1),
            'avg_conversations_per_user': self.avg_conversations_per_user,
            'avg_interactions_per_user': self.avg_interactions_per_user,
            'embedding_dim': self.embedding_dim
        }
    
    def print_statistics(self, verbose: bool = True):
        """Print dataset statistics. Set verbose=False khi DDP để chỉ rank 0 in."""
        if not verbose:
            return
        stats = self.get_statistics()
        print("=" * 50)
        print("Dataset Statistics:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50)


def collate_users(batch: List[UserData]) -> UserBatch:
    """
    Collate function for DataLoader.
    """
    return UserBatch(batch)

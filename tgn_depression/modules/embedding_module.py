"""
Embedding Module for TGN.

Compute node embeddings using temporal graph operations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    from .memory import Memory
except ImportError:
    from memory import Memory


class TimeEncode(nn.Module):
    """
    Time encoding using cosine functions (Time2Vec).
    
    Encodes time differences into learnable feature vectors.
    """
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: Output dimension of time encoding
        """
        super(TimeEncode, self).__init__()
        
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        
        # Initialize with frequencies spanning different time scales
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
            .float().reshape(dimension, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time values.
        
        Args:
            t: Time tensor [batch_size, seq_len] or [batch_size]
        
        Returns:
            Time encoding [batch_size, seq_len, dimension] or [batch_size, dimension]
        """
        # Add dimension for linear layer
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # Apply linear transformation and cosine
        output = torch.cos(self.w(t))
        
        return output


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer.
    
    Computes attention over temporal neighbors to aggregate information.
    """
    
    def __init__(self,
                 n_node_features: int,
                 n_edge_features: int,
                 n_time_features: int,
                 n_head: int = 2,
                 dropout: float = 0.1,
                 output_dimension: Optional[int] = None):
        """
        Args:
            n_node_features: Node feature dimension
            n_edge_features: Edge feature dimension
            n_time_features: Time encoding dimension
            n_head: Number of attention heads
            dropout: Dropout probability
            output_dimension: Output dimension (default: n_node_features)
        """
        super(TemporalAttentionLayer, self).__init__()
        
        self.n_head = n_head
        output_dimension = output_dimension or n_node_features
        
        # Query dimension: node features + time encoding
        self.query_dim = n_node_features + n_time_features
        
        # Key dimension: neighbor features + edge features + time encoding
        self.key_dim = n_node_features + n_edge_features + n_time_features
        
        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.key_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False
        )
        
        # Output projection: merge attention output with original features
        self.merger = nn.Sequential(
            nn.Linear(self.query_dim + n_node_features, output_dimension),
            nn.ReLU(),
            nn.Linear(output_dimension, output_dimension)
        )
    
    def forward(self,
                src_node_features: torch.Tensor,
                src_time_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_time_features: torch.Tensor,
                edge_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            src_node_features: [batch_size, n_node_features]
            src_time_features: [batch_size, 1, n_time_features]
            neighbor_features: [batch_size, n_neighbors, n_node_features]
            neighbor_time_features: [batch_size, n_neighbors, n_time_features]
            edge_features: [batch_size, n_neighbors, n_edge_features]
            neighbor_mask: [batch_size, n_neighbors] - True for padded positions
        
        Returns:
            output: [batch_size, output_dimension]
            attention_weights: [batch_size, n_neighbors]
        """
        # Build query: [1, batch_size, query_dim]
        src_node_features_expanded = src_node_features.unsqueeze(1)  # [batch, 1, node_feat]
        query = torch.cat([src_node_features_expanded, src_time_features], dim=2)
        query = query.permute(1, 0, 2)  # [1, batch, query_dim]
        
        # Build key/value: [n_neighbors, batch_size, key_dim]
        key = torch.cat([neighbor_features, edge_features, neighbor_time_features], dim=2)
        key = key.permute(1, 0, 2)  # [n_neighbors, batch, key_dim]
        
        # Handle nodes with no valid neighbors
        invalid_neighborhood_mask = neighbor_mask.all(dim=1, keepdim=True)
        neighbor_mask_copy = neighbor_mask.clone()
        neighbor_mask_copy[invalid_neighborhood_mask.squeeze(), 0] = False
        
        # Multi-head attention
        attn_output, attn_weights = self.multi_head_attention(
            query=query,
            key=key,
            value=key,
            key_padding_mask=neighbor_mask_copy
        )
        
        # attn_output: [1, batch, query_dim] -> [batch, query_dim]
        attn_output = attn_output.squeeze(0)
        attn_weights = attn_weights.squeeze(1) if attn_weights.dim() == 3 else attn_weights
        
        # Mask output for nodes with no neighbors
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        
        # Merge with original node features
        merged = torch.cat([attn_output, src_node_features], dim=1)
        output = self.merger(merged)
        
        return output, attn_weights


class EmbeddingModule(nn.Module):
    """
    Base class for embedding modules.
    """
    
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_features: torch.Tensor,
                 memory: Optional[Memory],
                 neighbor_finder,
                 time_encoder: TimeEncode,
                 n_layers: int,
                 n_node_features: int,
                 n_edge_features: int,
                 n_time_features: int,
                 embedding_dimension: int,
                 device: torch.device,
                 dropout: float = 0.1):
        super(EmbeddingModule, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.dropout = dropout
    
    def compute_embedding(self,
                          memory: torch.Tensor,
                          source_nodes: np.ndarray,
                          timestamps: np.ndarray,
                          n_layers: int,
                          n_neighbors: int = 20,
                          time_diffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class IdentityEmbedding(EmbeddingModule):
    """
    Identity embedding - just returns memory as embedding.
    
    z_i(t) = s_i(t)
    """
    
    def compute_embedding(self,
                          memory: torch.Tensor,
                          source_nodes: np.ndarray,
                          timestamps: np.ndarray,
                          n_layers: int,
                          n_neighbors: int = 20,
                          time_diffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return memory directly as embedding."""
        return memory[source_nodes, :]


class GraphEmbedding(EmbeddingModule):
    """
    Base class for graph-based embedding modules.
    
    Aggregates information from temporal neighbors.
    """
    
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_features: torch.Tensor,
                 memory: Optional[Memory],
                 neighbor_finder,
                 time_encoder: TimeEncode,
                 n_layers: int,
                 n_node_features: int,
                 n_edge_features: int,
                 n_time_features: int,
                 embedding_dimension: int,
                 device: torch.device,
                 n_heads: int = 2,
                 dropout: float = 0.1,
                 use_memory: bool = True):
        super(GraphEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, dropout
        )
        
        self.use_memory = use_memory
        self.n_heads = n_heads
    
    def compute_embedding(self,
                          memory: torch.Tensor,
                          source_nodes: np.ndarray,
                          timestamps: np.ndarray,
                          n_layers: int,
                          n_neighbors: int = 20,
                          time_diffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute embeddings using recursive graph attention.
        
        Args:
            memory: Current memory tensor [n_nodes, memory_dim]
            source_nodes: Node ids to compute embeddings for
            timestamps: Timestamps for each node
            n_layers: Number of attention layers
            n_neighbors: Number of neighbors to sample
            time_diffs: Optional time differences
        
        Returns:
            Node embeddings [batch_size, embedding_dim]
        """
        assert n_layers >= 0
        
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.from_numpy(timestamps).float().to(self.device)
        
        # Time encoding for source nodes (time span = 0)
        source_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
        if source_time_embedding.dim() == 2:
            source_time_embedding = source_time_embedding.unsqueeze(1)
        
        # Base features: node_features + memory (if available)
        source_node_features = self.node_features[source_nodes_torch, :]
        
        if self.use_memory and memory is not None:
            memory_features = memory[source_nodes, :]
            source_node_features = memory_features + source_node_features
        
        # Base case: no graph aggregation
        if n_layers == 0:
            return source_node_features
        
        # Recursive case: aggregate from neighbors
        # First, get embeddings at layer n-1 for source nodes
        source_node_conv_embeddings = self.compute_embedding(
            memory, source_nodes, timestamps,
            n_layers=n_layers - 1, n_neighbors=n_neighbors
        )
        
        # Sample temporal neighbors
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
            source_nodes, timestamps, n_neighbors=n_neighbors
        )
        
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
        
        # Time differences: current_time - edge_time
        edge_deltas = timestamps[:, np.newaxis] - edge_times
        edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
        
        # Get neighbor embeddings (flatten, compute, reshape)
        neighbors_flat = neighbors.flatten()
        timestamps_repeated = np.repeat(timestamps, n_neighbors)
        
        neighbor_embeddings = self.compute_embedding(
            memory, neighbors_flat, timestamps_repeated,
            n_layers=n_layers - 1, n_neighbors=n_neighbors
        )
        
        # Reshape: [batch * n_neighbors, dim] -> [batch, n_neighbors, dim]
        effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbor_embeddings = neighbor_embeddings.view(
            len(source_nodes), effective_n_neighbors, -1
        )
        
        # Time encoding for edges
        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        
        # Edge features
        edge_features = self.edge_features[edge_idxs_torch, :]
        
        # Create mask for padding (neighbor_id = 0 means no neighbor)
        neighbor_mask = neighbors_torch == 0
        
        # Aggregate from neighbors
        source_embedding = self.aggregate(
            n_layers,
            source_node_conv_embeddings,
            source_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            edge_features,
            neighbor_mask
        )
        
        return source_embedding
    
    def aggregate(self,
                  n_layer: int,
                  source_node_features: torch.Tensor,
                  source_time_embedding: torch.Tensor,
                  neighbor_embeddings: torch.Tensor,
                  edge_time_embeddings: torch.Tensor,
                  edge_features: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GraphAttentionEmbedding(GraphEmbedding):
    """
    Graph attention embedding module.
    
    Uses multi-head attention to aggregate from temporal neighbors.
    """
    
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_features: torch.Tensor,
                 memory: Optional[Memory],
                 neighbor_finder,
                 time_encoder: TimeEncode,
                 n_layers: int,
                 n_node_features: int,
                 n_edge_features: int,
                 n_time_features: int,
                 embedding_dimension: int,
                 device: torch.device,
                 n_heads: int = 2,
                 dropout: float = 0.1,
                 use_memory: bool = True):
        super(GraphAttentionEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, n_heads,
            dropout, use_memory
        )
        
        # Attention layers for each graph layer
        self.attention_layers = nn.ModuleList([
            TemporalAttentionLayer(
                n_node_features=n_node_features,
                n_edge_features=n_edge_features,
                n_time_features=n_time_features,
                n_head=n_heads,
                dropout=dropout,
                output_dimension=n_node_features
            )
            for _ in range(n_layers)
        ])
    
    def aggregate(self,
                  n_layer: int,
                  source_node_features: torch.Tensor,
                  source_time_embedding: torch.Tensor,
                  neighbor_embeddings: torch.Tensor,
                  edge_time_embeddings: torch.Tensor,
                  edge_features: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """Aggregate using temporal attention."""
        attention_layer = self.attention_layers[n_layer - 1]
        
        output, _ = attention_layer(
            src_node_features=source_node_features,
            src_time_features=source_time_embedding,
            neighbor_features=neighbor_embeddings,
            neighbor_time_features=edge_time_embeddings,
            edge_features=edge_features,
            neighbor_mask=mask
        )
        
        return output


class GraphSumEmbedding(GraphEmbedding):
    """
    Graph sum embedding module.
    
    Simple sum aggregation - faster than attention.
    """
    
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_features: torch.Tensor,
                 memory: Optional[Memory],
                 neighbor_finder,
                 time_encoder: TimeEncode,
                 n_layers: int,
                 n_node_features: int,
                 n_edge_features: int,
                 n_time_features: int,
                 embedding_dimension: int,
                 device: torch.device,
                 n_heads: int = 2,
                 dropout: float = 0.1,
                 use_memory: bool = True):
        super(GraphSumEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, n_heads,
            dropout, use_memory
        )
        
        # Linear layers for aggregation
        self.linear_1 = nn.ModuleList([
            nn.Linear(n_node_features + n_time_features + n_edge_features, n_node_features)
            for _ in range(n_layers)
        ])
        self.linear_2 = nn.ModuleList([
            nn.Linear(n_node_features + n_node_features + n_time_features, n_node_features)
            for _ in range(n_layers)
        ])
    
    def aggregate(self,
                  n_layer: int,
                  source_node_features: torch.Tensor,
                  source_time_embedding: torch.Tensor,
                  neighbor_embeddings: torch.Tensor,
                  edge_time_embeddings: torch.Tensor,
                  edge_features: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """Aggregate using sum."""
        # Combine neighbor info
        neighbors_combined = torch.cat([
            neighbor_embeddings, edge_time_embeddings, edge_features
        ], dim=2)
        
        # Transform and sum
        neighbor_transformed = self.linear_1[n_layer - 1](neighbors_combined)
        
        # Mask padded neighbors
        neighbor_transformed = neighbor_transformed.masked_fill(mask.unsqueeze(-1), 0)
        
        # Sum aggregation
        neighbors_sum = torch.relu(torch.sum(neighbor_transformed, dim=1))
        
        # Combine with source features
        source_time_embedding_squeezed = source_time_embedding.squeeze(1)
        combined = torch.cat([
            neighbors_sum, source_node_features, source_time_embedding_squeezed
        ], dim=1)
        
        output = self.linear_2[n_layer - 1](combined)
        
        return output


def get_embedding_module(
    module_type: str,
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    memory: Optional[Memory],
    neighbor_finder,
    time_encoder: TimeEncode,
    n_layers: int,
    n_node_features: int,
    n_edge_features: int,
    n_time_features: int,
    embedding_dimension: int,
    device: torch.device,
    n_heads: int = 2,
    dropout: float = 0.1,
    use_memory: bool = True,
    n_neighbors: Optional[int] = None
) -> EmbeddingModule:
    """
    Factory function to create embedding module.
    
    Args:
        module_type: 'graph_attention', 'graph_sum', or 'identity'
        ... other args passed to constructor
    
    Returns:
        EmbeddingModule instance
    """
    if module_type == 'graph_attention':
        return GraphAttentionEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory
        )
    elif module_type == 'graph_sum':
        return GraphSumEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory
        )
    elif module_type == 'identity':
        return IdentityEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown embedding module type: {module_type}")

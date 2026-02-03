"""
Neighbor Finder for Temporal Graph.

Tìm temporal neighbors của một node tại một thời điểm cụ thể.
Hỗ trợ tối ưu L-hop ego subgraph (layer=0, 1, 2) để tránh tính toán thừa.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from collections import defaultdict


class NeighborFinder:
    """
    Finds temporal neighbors for nodes in a dynamic graph.
    
    Neighbors are users who have interacted with the target user
    BEFORE a given timestamp.
    """
    
    def __init__(self, 
                 adj_list: List[List[Tuple[int, int, float]]],
                 uniform: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            adj_list: List where adj_list[i] contains tuples of 
                     (neighbor_id, edge_idx, timestamp) for node i
            uniform: If True, sample uniformly; if False, sample most recent
            seed: Random seed for reproducibility
        """
        self.uniform = uniform
        
        # Sort neighbors by timestamp for each node
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        
        for neighbors in adj_list:
            # Sort by timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))
        
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
    
    def find_before(self, src_idx: int, cut_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find all interactions of node src_idx happening before cut_time.
        """
        if src_idx >= len(self.node_to_edge_timestamps):
            return np.array([]), np.array([]), np.array([])
        
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        
        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i]
        )
    
    def get_temporal_neighbor(self,
                              source_nodes: np.ndarray,
                              timestamps: np.ndarray,
                              n_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get temporal neighbors for a batch of source nodes.
        """
        assert len(source_nodes) == len(timestamps)
        
        n_neighbors = max(n_neighbors, 1)
        batch_size = len(source_nodes)
        
        neighbors = np.zeros((batch_size, n_neighbors), dtype=np.int32)
        edge_times = np.zeros((batch_size, n_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((batch_size, n_neighbors), dtype=np.int32)
        
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            src_neighbors, src_edge_idxs, src_edge_times = self.find_before(
                source_node, timestamp
            )
            
            if len(src_neighbors) == 0:
                continue
            
            if self.uniform:
                if len(src_neighbors) > n_neighbors:
                    sampled_idx = self.random_state.choice(
                        len(src_neighbors), n_neighbors, replace=False
                    )
                else:
                    sampled_idx = np.arange(len(src_neighbors))
                
                sampled_idx = sampled_idx[np.argsort(src_edge_times[sampled_idx])]
                
                n_sampled = len(sampled_idx)
                neighbors[i, n_neighbors - n_sampled:] = src_neighbors[sampled_idx]
                edge_times[i, n_neighbors - n_sampled:] = src_edge_times[sampled_idx]
                edge_idxs[i, n_neighbors - n_sampled:] = src_edge_idxs[sampled_idx]
            else:
                n_take = min(len(src_neighbors), n_neighbors)
                
                neighbors[i, n_neighbors - n_take:] = src_neighbors[-n_take:]
                edge_times[i, n_neighbors - n_take:] = src_edge_times[-n_take:]
                edge_idxs[i, n_neighbors - n_take:] = src_edge_idxs[-n_take:]
        
        return neighbors, edge_idxs, edge_times


def get_temporal_ego_nodes(
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    target_user: int,
    end_time: float,
    n_layers: int,
) -> Set[int]:
    """
    Tính tập node nằm trong L-hop temporal ego subgraph của target_user.
    Chỉ hỗ trợ n_layers in (0, 1, 2). Nếu n_layers > 2 trả về set rỗng (caller dùng full graph).

    Args:
        sources, destinations, timestamps: Cạnh và thời gian
        target_user: Node trung tâm
        end_time: Chỉ xét cạnh có timestamp <= end_time
        n_layers: Số layer (0 = chỉ target_user, 1 = + 1-hop, 2 = + 2-hop)

    Returns:
        Set các node id trong ego subgraph.
    """
    if n_layers < 0:
        return set()
    ego: Set[int] = {target_user}
    if n_layers == 0:
        return ego
    # Chỉ tối ưu cho 1, 2 layer
    n_hops = min(int(n_layers), 2)
    for _ in range(n_hops):
        next_ego = set(ego)
        for i in range(len(sources)):
            if timestamps[i] > end_time:
                continue
            s, d = int(sources[i]), int(destinations[i])
            if s in ego:
                next_ego.add(d)
            if d in ego:
                next_ego.add(s)
        ego = next_ego
    return ego


def get_temporal_ego_subgraph(
    sources: np.ndarray,
    destinations: np.ndarray,
    edge_idxs: np.ndarray,
    timestamps: np.ndarray,
    target_user: int,
    end_time: float,
    n_layers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lọc các cạnh thuộc L-hop temporal ego subgraph của target_user, giữ thứ tự thời gian.
    Chỉ hỗ trợ n_layers in (0, 1, 2). Nếu n_layers > 2 trả về toàn bộ cạnh (không lọc).

    Returns:
        (sources_f, destinations_f, edge_idxs_f, timestamps_f) đã sort theo timestamps_f.
    """
    if n_layers < 0 or n_layers > 2:
        return sources, destinations, edge_idxs, timestamps
    ego = get_temporal_ego_nodes(sources, destinations, timestamps, target_user, end_time, n_layers)
    # Cạnh cần giữ: timestamp <= end_time và ít nhất một đầu mút trong ego (để cập nhật memory và neighbor_finder)
    mask = np.zeros(len(sources), dtype=bool)
    for i in range(len(sources)):
        if timestamps[i] <= end_time and (int(sources[i]) in ego or int(destinations[i]) in ego):
            mask[i] = True
    if not np.any(mask):
        return (
            np.array([], dtype=sources.dtype),
            np.array([], dtype=destinations.dtype),
            np.array([], dtype=edge_idxs.dtype),
            np.array([], dtype=timestamps.dtype),
        )
    s_f = sources[mask]
    d_f = destinations[mask]
    e_f = edge_idxs[mask]
    t_f = timestamps[mask]
    order = np.argsort(t_f)
    return s_f[order], d_f[order], e_f[order], t_f[order]


def get_neighbor_finder(
    sources: np.ndarray,
    destinations: np.ndarray,
    edge_idxs: np.ndarray,
    timestamps: np.ndarray,
    n_nodes: int,
    uniform: bool = False
) -> NeighborFinder:
    """
    Create a NeighborFinder from interaction data.
    """
    adj_list = [[] for _ in range(n_nodes)]
    
    for source, dest, edge_idx, ts in zip(sources, destinations, edge_idxs, timestamps):
        adj_list[source].append((dest, edge_idx, ts))
        adj_list[dest].append((source, edge_idx, ts))
    
    return NeighborFinder(adj_list, uniform=uniform)

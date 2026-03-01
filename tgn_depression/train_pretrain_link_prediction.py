"""
Pretrain TGN encoder bằng link prediction (giống TGN gốc).

- Xây global edge stream từ toàn bộ users (flatten conversations, sort by time).
- Mỗi batch: (sources, destinations, timestamps, edge_idxs) + negative sampling.
- Loss: BCE(pos_score, 1) + BCE(neg_score, 0).
- Lưu checkpoint encoder; load vào TGNUserSequence với strict=False (bỏ affinity_score).

Pipeline đúng TGN:
  1. Pretrain: python -m tgn_depression.train_pretrain_link_prediction --data_dir /path/to/data
  2. Supervised: python -m tgn_depression.train_supervised_tgn_style --data_dir /path/to/data --encoder_checkpoint ./saved_models_pretrain/encoder_link_pred.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.tgn_link_prediction import TGNLinkPrediction
from utils.data_structures import UserData
from utils.data_loader import (
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import set_seed, get_device
from utils.neighbor_finder import get_neighbor_finder


def build_global_edge_stream(
    users: List[UserData],
    only_edges_with_target_user: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gom interaction từ các user thành một stream (sources, destinations, timestamps, edge_idxs),
    sắp xếp theo thời gian. edge_idxs = post_id (index vào post_embeddings).

    Nếu only_edges_with_target_user=True: chỉ giữ cạnh có ít nhất 1 endpoint là target user
    (user có label trong dataset).
    """
    target_user_ids = {u.user_id for u in users} if only_edges_with_target_user else None
    rows = []
    for u in users:
        for conv in u.get_conversations_sorted():
            if conv.n_interactions == 0:
                continue
            for i in range(len(conv.source_users)):
                src, dst = int(conv.source_users[i]), int(conv.dest_users[i])
                if target_user_ids is not None and (src not in target_user_ids and dst not in target_user_ids):
                    continue
                rows.append((
                    src,
                    dst,
                    float(conv.timestamps[i]),
                    int(conv.post_ids[i]),
                ))
    if not rows:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )
    rows.sort(key=lambda r: r[2])
    sources = np.array([r[0] for r in rows], dtype=np.int64)
    destinations = np.array([r[1] for r in rows], dtype=np.int64)
    timestamps = np.array([r[2] for r in rows], dtype=np.float64)
    edge_idxs = np.array([r[3] for r in rows], dtype=np.int64)
    return sources, destinations, timestamps, edge_idxs


class RandEdgeSampler:
    """Negative sampling: sample random destination cho mỗi source (giống TGN)."""

    def __init__(self, src_list: np.ndarray, dst_list: np.ndarray, seed: int = 0):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        si = self.rng.randint(0, len(self.src_list), size)
        di = self.rng.randint(0, len(self.dst_list), size)
        return self.src_list[si], self.dst_list[di]


class EdgeStreamDataset(Dataset):
    """Dataset cho từng batch edge (index = batch_start)."""

    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_idxs: np.ndarray,
        batch_size: int,
    ):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.batch_size = batch_size
        self.n_batches = (len(sources) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.sources))
        return (
            self.sources[start:end],
            self.destinations[start:end],
            self.timestamps[start:end],
            self.edge_idxs[start:end],
        )


def train_epoch(
    model: TGNLinkPrediction,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    neg_sampler: RandEdgeSampler,
    device: torch.device,
    n_neighbors: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        sources, destinations, timestamps, edge_idxs = batch
        sources = sources.numpy()
        destinations = destinations.numpy()
        timestamps = timestamps.numpy()
        edge_idxs = edge_idxs.numpy()
        size = len(sources)
        neg_src, neg_dst = neg_sampler.sample(size)

        optimizer.zero_grad()
        pos_score, neg_score = model.compute_edge_probabilities(
            sources, destinations, neg_dst, timestamps, edge_idxs, n_neighbors=n_neighbors
        )
        pos_loss = nn.functional.binary_cross_entropy(pos_score, torch.ones_like(pos_score, device=device))
        neg_loss = nn.functional.binary_cross_entropy(neg_score, torch.zeros_like(neg_score, device=device))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * size
        n += size
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Pretrain TGN encoder by link prediction")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--neg_folder", type=str, default="neg")
    parser.add_argument("--pos_folder", type=str, default="pos")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--n_total_users", type=int, default=100)
    parser.add_argument("--n_target_users", type=int, default=50)
    parser.add_argument("--n_conversations", type=int, default=200)
    parser.add_argument("--avg_interactions", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--split_method", type=str, default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_ego_hops", type=int, default=-1, help="<0 = không lọc ego")

    parser.add_argument("--memory_dim", type=int, default=172)
    parser.add_argument("--n_ego_layers", type=int, default=2)
    parser.add_argument("--embedding_module_type", type=str, default="graph_attention", choices=["identity", "graph_attention", "graph_sum"])
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--message_function", type=str, default="identity", choices=["identity", "mlp"])
    parser.add_argument("--aggregator", type=str, default="last")
    parser.add_argument("--memory_update_at_start", action="store_true", default=True, help="Update memory at start of batch (TGN default)")
    parser.add_argument("--no_memory_update_at_start", action="store_false", dest="memory_update_at_start")
    parser.add_argument("--only_target_user_edges", action="store_true", default=True, help="Pretrain chỉ với cạnh có ≥1 node là target user (mặc định)")
    parser.add_argument("--no_only_target_user_edges", action="store_false", dest="only_target_user_edges")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="./saved_models_pretrain")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.use_dummy_data:
        train_dataset, val_dataset, test_dataset, metadata = create_dummy_data(
            n_total_users=args.n_total_users,
            n_target_users=args.n_target_users,
            n_conversations=args.n_conversations,
            avg_interactions=args.avg_interactions,
            embedding_dim=args.embedding_dim,
            depression_ratio=0.3,
            save_dir=None,
        )
    else:
        train_dataset, val_dataset, test_dataset, metadata = load_depression_data_from_parquet_folders(
            data_dir=args.data_dir,
            neg_folder=args.neg_folder,
            pos_folder=args.pos_folder,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_method=args.split_method,
            seed=args.seed,
            max_ego_hops=None if args.max_ego_hops < 0 else args.max_ego_hops,
            verbose=True,
        )

    for ds in (train_dataset, val_dataset):
        ds.users = [u for u in ds.users if u.total_interactions > 0]
        for u in ds.users:
            u.conversations = [c for c in u.conversations if c.n_interactions > 0]
        ds.users = [u for u in ds.users if u.total_interactions > 0]

    sources, destinations, timestamps, edge_idxs = build_global_edge_stream(
        train_dataset.users, only_edges_with_target_user=args.only_target_user_edges
    )
    if len(sources) == 0:
        raise ValueError("No edges in train set")
    n_nodes = metadata["n_total_users"]
    print(f"Global edge stream: {len(sources)} edges, n_nodes={n_nodes}" + (
        " (chỉ cạnh có ≥1 node là target user)" if args.only_target_user_edges else ""
    ))

    neighbor_finder = get_neighbor_finder(sources, destinations, edge_idxs, timestamps, n_nodes, uniform=False)
    neg_sampler = RandEdgeSampler(sources, destinations, seed=args.seed)

    node_features = np.zeros((n_nodes, args.memory_dim), dtype=np.float32)
    edge_features = train_dataset.post_embeddings

    model = TGNLinkPrediction(
        neighbor_finder=neighbor_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_ego_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        use_memory=True,
        memory_update_at_start=args.memory_update_at_start,
        message_dimension=100,
        memory_dimension=args.memory_dim,
        embedding_module_type=args.embedding_module_type,
        message_function_type=args.message_function,
        aggregator_type=args.aggregator,
        n_neighbors=args.n_neighbors,
    ).to(device)

    def _collate(batch):
        s, d, t, e = batch[0]
        return (
            torch.from_numpy(s),
            torch.from_numpy(d),
            torch.from_numpy(t),
            torch.from_numpy(e),
        )

    dataset = EdgeStreamDataset(sources, destinations, timestamps, edge_idxs, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.memory.reset_state()
        t0 = time.time()
        loss = train_epoch(model, dataloader, optimizer, neg_sampler, device, args.n_neighbors)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) link_pred_loss={loss:.4f}")

    encoder_path = f"{args.save_dir}/encoder_link_pred.pth"
    torch.save(model.state_dict(), encoder_path)
    print(f"Saved encoder to {encoder_path} (load vào TGNUserSequence với strict=False)")
    with open(f"{args.save_dir}/pretrain_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()

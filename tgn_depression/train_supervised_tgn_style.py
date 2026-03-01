"""
Train theo đúng cách TGN gốc (tgn/train_supervised.py):

1. Encoder (TGNUserSequence) = frozen, dùng để tính embedding user.
2. Chỉ train Decoder (ClassificationHead): với mỗi user, lấy embedding qua encoder (no_grad),
   rồi classifier(embedding) -> loss -> backward chỉ trên decoder.

Cách chạy:
  # Encoder random (chưa pretrain), chỉ train decoder:
  python -m tgn_depression.train_supervised_tgn_style --data_dir /path/to/data

  # Load encoder từ checkpoint (vd. từ train.py end-to-end), rồi train decoder:
  python -m tgn_depression.train_supervised_tgn_style --data_dir /path/to/data --encoder_checkpoint ./saved_models/best_model.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_user_sequence import TGNUserSequence
from utils.data_structures import DepressionDataset, UserData
from utils.data_loader import (
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, compute_class_weights


class UserListDataset(Dataset):
    """Dataset bọc list UserData cho DataLoader."""

    def __init__(self, users: List[UserData]):
        self.users = users

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> UserData:
        return self.users[idx]


def _collate_users(batch: List[UserData]) -> List[UserData]:
    """Collate list of UserData (batch)."""
    return list(batch)


def get_user_embedding(encoder: TGNUserSequence, user_data: UserData, device: torch.device, memory_dim: int) -> torch.Tensor:
    """
    Chạy encoder (no_grad) để lấy embedding user tại thời điểm cuối chuỗi conversations.
    Trả về [memory_dim]. Nếu user không có interaction thì trả về zeros.
    """
    if user_data.total_interactions == 0:
        return torch.zeros(memory_dim, device=device, dtype=torch.float32)
    encoder.reset_state()
    with torch.no_grad():
        result = encoder.forward(
            user_data,
            return_per_event=False,
            return_logits=False,
        )
    if not result:
        return torch.zeros(memory_dim, device=device, dtype=torch.float32)
    return result[-1].detach()


def train_epoch_tgn_style(
    encoder: TGNUserSequence,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    memory_dim: int,
) -> Tuple[float, Dict]:
    """Một epoch: lấy embedding từ encoder (eval, no_grad), train decoder."""
    encoder.eval()
    decoder.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n = 0

    for batch_users in dataloader:
        if not batch_users:
            continue
        embeddings = []
        labels = []
        for user_data in batch_users:
            emb = get_user_embedding(encoder, user_data, device, memory_dim)
            embeddings.append(emb)
            labels.append(user_data.label)
        embeddings = torch.stack(embeddings)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = decoder(embeddings)
        loss = criterion(logits, labels_t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += len(labels)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs.astype(np.float32))
        all_labels.extend(labels)

    avg_loss = total_loss / max(n, 1)
    metrics = {}
    if all_preds:
        all_preds = np.array(all_preds, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        pred_labels = (all_preds > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(all_labels, pred_labels)
        if len(np.unique(all_labels)) > 1:
            metrics["auc"] = roc_auc_score(all_labels, all_preds)
            metrics["f1"] = f1_score(all_labels, pred_labels)
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels, pred_labels, average="binary", zero_division=0
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
    return avg_loss, metrics


def evaluate_tgn_style(
    encoder: TGNUserSequence,
    decoder: nn.Module,
    dataset: DepressionDataset,
    device: torch.device,
    memory_dim: int,
    batch_size: int = 32,
) -> Tuple[float, Dict]:
    """Eval: embedding từ encoder (no_grad), decoder -> metrics."""
    encoder.eval()
    decoder.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(0, len(dataset.users), batch_size):
            batch_users = dataset.users[i : i + batch_size]
            embeddings = []
            labels = []
            for user_data in batch_users:
                emb = get_user_embedding(encoder, user_data, device, memory_dim)
                embeddings.append(emb)
                labels.append(user_data.label)
            embeddings = torch.stack(embeddings)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            logits = decoder(embeddings)
            loss = criterion(logits, labels_t)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs.astype(np.float32))
            all_labels.extend(labels)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    avg_loss = total_loss / max(len(all_labels), 1)
    pred_labels = (all_preds > 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(all_labels, pred_labels)}
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, pred_labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average="binary", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train decoder only (TGN style): encoder frozen, chỉ train classifier."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa neg/ pos/ parquet")
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
    parser.add_argument("--max_ego_hops", type=int, default=2)

    parser.add_argument("--memory_dim", type=int, default=172)
    parser.add_argument("--n_ego_layers", type=int, default=2)
    parser.add_argument("--embedding_module_type", type=str, default="graph_attention", choices=["identity", "graph_attention", "graph_sum"])
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--encoder_checkpoint", type=str, default=None, help="Path checkpoint encoder (TGN). Nếu None thì encoder random, vẫn freeze.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR cho decoder")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (số user) cho decoder")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./saved_models_supervised")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load data
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

    # Drop empty conversations / users
    for ds in (train_dataset, val_dataset):
        ds.users = [u for u in ds.users if u.total_interactions > 0]
        for u in ds.users:
            u.conversations = [c for c in u.conversations if c.n_interactions > 0]
        ds.users = [u for u in ds.users if u.total_interactions > 0]

    print(f"Train users: {len(train_dataset.users)}, Val users: {len(val_dataset.users)}")

    # Build full model (encoder + classifier)
    model = TGNUserSequence(
        n_users=metadata["n_total_users"],
        edge_features=train_dataset.post_embeddings,
        device=device,
        memory_dimension=args.memory_dim,
        n_ego_layers=args.n_ego_layers,
        embedding_module_type=args.embedding_module_type,
        n_heads=args.n_heads,
        n_neighbors=args.n_neighbors,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    # Load encoder checkpoint (từ pretrain link prediction hoặc từ train.py)
    if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
        state = torch.load(args.encoder_checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded encoder from {args.encoder_checkpoint} (strict=False, bỏ affinity_score/classifier không khớp)")

    # Freeze encoder: tất cả trừ classifier
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}")

    # DataLoader: batch users
    train_loader = DataLoader(
        UserListDataset(train_dataset.users),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_users,
    )

    train_labels = np.array([u.label for u in train_dataset.users])
    class_weights = compute_class_weights(train_labels).to(device).float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(decoder_params, lr=args.lr, weight_decay=args.weight_decay)
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)

    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        # Init memory mỗi epoch (giống TGN train_supervised)
        model.memory.reset_state()
        train_loss, train_metrics = train_epoch_tgn_style(
            model, model.classifier, train_loader, optimizer, criterion, device, args.memory_dim
        )
        val_loss, val_metrics = evaluate_tgn_style(
            model, model.classifier, val_dataset, device, args.memory_dim, batch_size=args.batch_size
        )
        val_auc = val_metrics.get("auc", 0.0)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) train_loss={train_loss:.4f} train_metrics={train_metrics} val_auc={val_auc:.4f} val_metrics={val_metrics}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.classifier.state_dict(), f"{args.save_dir}/decoder_best.pth")

        if early_stopper.early_stop_check(val_auc):
            print(f"Early stop at epoch {epoch+1}")
            break

    # Load best decoder
    model.classifier.load_state_dict(torch.load(f"{args.save_dir}/decoder_best.pth", map_location=device))
    test_loss, test_metrics = evaluate_tgn_style(
        model, model.classifier, test_dataset, device, args.memory_dim, batch_size=args.batch_size
    )
    print(f"Test metrics: {test_metrics}")

    results = {"best_epoch": best_epoch, "best_val_auc": best_val_auc, "test_metrics": test_metrics, "args": vars(args)}
    with open(f"{args.save_dir}/results_supervised.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/results_supervised.json")


if __name__ == "__main__":
    main()

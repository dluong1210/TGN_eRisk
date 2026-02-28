"""
Training script for TGNUserSequence (eRisk).

Sliding-window training: mỗi window của conversation embeddings → aggregate → classifier → loss.
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_user_sequence import TGNUserSequence
from utils.data_structures import DepressionDataset
from utils.data_loader import (
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, compute_class_weights


def sliding_windows(embeddings: List[torch.Tensor], window_size: int) -> List[torch.Tensor]:
    """Tạo các window từ chuỗi embeddings. Mỗi window = [emb_i, ..., emb_{i+W-1}]."""
    if len(embeddings) == 0:
        return []
    if window_size >= len(embeddings):
        return [torch.stack(embeddings)]
    windows = []
    for i in range(len(embeddings) - window_size + 1):
        w = torch.stack(embeddings[i : i + window_size])
        windows.append(w)
    return windows


def aggregate_window(window: torch.Tensor, mode: str = "last") -> torch.Tensor:
    """Aggregate embeddings trong window: last (lấy cuối) hoặc mean."""
    if mode == "last":
        return window[-1:].squeeze(0)  # [dim]
    if mode == "mean":
        return window.mean(dim=0)
    raise ValueError(f"Unknown aggregation: {mode}")


def setup_logging(log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/train_{int(time.time())}.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def train_epoch(
    model: TGNUserSequence,
    dataset: DepressionDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    window_size: int,
    window_aggregation: str,
    max_grad_norm: float = 0.0,
) -> Tuple[float, Dict]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_samples = 0

    for user_data in dataset.users:
        if user_data.total_interactions == 0:
            continue

        optimizer.zero_grad()
        model.reset_state()
        embeddings = model.forward(user_data, return_per_event=False)

        if len(embeddings) == 0:
            continue

        windows = sliding_windows(embeddings, window_size)
        if len(windows) == 0:
            continue

        batch_agg = []
        for w in windows:
            h = aggregate_window(w, window_aggregation)
            batch_agg.append(h)
        batch_agg = torch.stack(batch_agg)

        logits = model.classifier(batch_agg)
        label = user_data.label
        labels_t = torch.full(
            (len(windows),), label, dtype=torch.long, device=device
        )
        loss = criterion(logits, labels_t)
        loss.backward()
        total_loss += loss.item()
        n_samples += len(windows)

        with torch.inference_mode():
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.append(probs.astype(np.float32))
        all_labels.append(np.full(len(windows), label, dtype=np.int64))

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / max(n_samples, 1)
    metrics = {}
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
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


def evaluate(
    model: TGNUserSequence,
    dataset: DepressionDataset,
    device: torch.device,
    window_size: int,
    window_aggregation: str,
) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for user_data in dataset.users:
            if user_data.total_interactions == 0:
                all_preds.append(0.5)
                all_labels.append(user_data.label)
                continue

            model.reset_state()
            embeddings = model.forward(user_data, return_per_event=False)

            if len(embeddings) == 0:
                all_preds.append(0.5)
                all_labels.append(user_data.label)
                continue

            windows = sliding_windows(embeddings, window_size)
            if len(windows) == 0:
                h = aggregate_window(torch.stack(embeddings), window_aggregation)
                logits = model.classifier(h.unsqueeze(0))
            else:
                batch_agg = torch.stack([
                    aggregate_window(w, window_aggregation) for w in windows
                ])
                logits = model.classifier(batch_agg)
                logits = logits[-1:]  # Lấy prediction từ window cuối

            label = user_data.label
            total_loss += criterion(logits, torch.tensor([label], dtype=torch.long, device=device)).item()
            all_preds.append(torch.softmax(logits, dim=1)[0, 1].item())
            all_labels.append(label)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    avg_loss = total_loss / max(len(all_labels), 1)
    metrics = {}
    if len(all_labels) > 0:
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

    return avg_loss, metrics, all_preds, all_labels


def main_worker(args, device=None):
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")

    set_seed(args.seed)
    if device is None:
        device = get_device(args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    if args.use_dummy_data:
        train_dataset, val_dataset, test_dataset, metadata = create_dummy_data(
            n_total_users=args.n_total_users,
            n_target_users=args.n_target_users,
            n_conversations=args.n_conversations,
            avg_interactions=args.avg_interactions,
            embedding_dim=args.embedding_dim,
            depression_ratio=0.3,
            save_dir=args.data_dir if args.save_dummy else None,
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
            max_ego_hops=None if getattr(args, "max_ego_hops", 2) < 0 else getattr(args, "max_ego_hops", 2),
            verbose=True,
        )

    def _drop_empty(ds: DepressionDataset) -> int:
        dropped = 0
        for u in ds.users:
            before = len(u.conversations)
            u.conversations = [c for c in u.conversations if c.n_interactions > 0]
            dropped += before - len(u.conversations)
        return dropped

    def _filter_nonempty(ds: DepressionDataset) -> DepressionDataset:
        ds.users = [u for u in ds.users if u.total_interactions > 0]
        return ds

    _drop_empty(train_dataset)
    _drop_empty(val_dataset)
    train_dataset = _filter_nonempty(train_dataset)
    val_dataset = _filter_nonempty(val_dataset)

    logger.info(f"Train: {len(train_dataset.users)} users")
    train_dataset.print_statistics(verbose=True)

    train_labels = np.array([u.label for u in train_dataset.users])
    class_weights = compute_class_weights(train_labels).to(device).float()

    logger.info("Initializing TGNUserSequence...")
    model = TGNUserSequence(
        n_users=metadata["n_total_users"],
        edge_features=train_dataset.post_embeddings,
        device=device,
        memory_dimension=args.memory_dim,
        n_ego_layers=args.n_ego_layers,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)

    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            window_size=args.window_size,
            window_aggregation=args.window_aggregation,
            max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        )
        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            dataset=val_dataset,
            device=device,
            window_size=args.window_size,
            window_aggregation=args.window_aggregation,
        )

        val_auc = val_metrics.get("auc", 0.0)
        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
            logger.info(f"  New best model saved! (AUC: {val_auc:.4f})")

        if early_stopper.early_stop_check(val_auc):
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

    best_path = f"{args.save_dir}/best_model.pth"
    if not Path(best_path).exists():
        torch.save(model.state_dict(), best_path)
    logger.info(f"Best model: {best_path} (epoch {best_epoch}, val AUC: {best_val_auc:.4f})")

    results = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "args": vars(args),
    }
    with open(f"{args.save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.save_dir}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Train TGNUserSequence for eRisk")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_format", type=str, default="parquet_folders")
    parser.add_argument("--neg_folder", type=str, default="neg")
    parser.add_argument("--pos_folder", type=str, default="pos")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--save_dummy", action="store_true")
    parser.add_argument("--n_total_users", type=int, default=100)
    parser.add_argument("--n_target_users", type=int, default=50)
    parser.add_argument("--n_conversations", type=int, default=200)
    parser.add_argument("--avg_interactions", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--split_method", type=str, default="stratified")

    parser.add_argument("--memory_dim", type=int, default=172)
    parser.add_argument("--n_ego_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--window_aggregation", type=str, default="last", choices=["last", "mean"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_ego_hops", type=int, default=2)

    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    args = parser.parse_args()
    main_worker(args)


if __name__ == "__main__":
    main()

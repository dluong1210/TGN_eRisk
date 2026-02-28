"""
Training script for TGNUserSequence (eRisk).

Sliding-window (chỉ positive user): split chuỗi conversations thành nhiều chuỗi nhỏ
→ mỗi chuỗi nhỏ feed vào model → 1 embedding → 1 sample (label=1). Augment data.
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_user_sequence import TGNUserSequence
from utils.data_structures import DepressionDataset, UserData
from utils.data_loader import (
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, compute_class_weights


def get_distributed_info():
    """Get rank, world_size, local_rank from env (set by torchrun)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def partition_users_by_rank(users: List, rank: int, world_size: int, seed: int = 42) -> List:
    """
    Chia danh sách users cho từng GPU (rank).
    Shuffle thống nhất (cùng seed) rồi chia: rank 0 lấy [0, world_size, 2*world_size, ...],
    rank 1 lấy [1, 1+world_size, ...]. Cân bằng số user giữa các GPU.
    """
    n = len(users)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    my_indices = indices[rank::world_size]
    return [users[i] for i in my_indices]


def sliding_windows_conversations(
    conversations: List,
    window_size: int,
    overlap: int = 0,
    min_window_size: int = 1,
) -> List[List]:
    """
    Split chuỗi conversations (input data) thành nhiều chuỗi nhỏ để augment.

    Chỉ dùng cho positive user. Mỗi chuỗi nhỏ = 1 training sample (label=1).
    step = window_size - overlap; window cuối có thể partial.

    Args:
        conversations: Danh sách Conversation (thứ tự thời gian)
        window_size: Số conversation mỗi window (e.g. 50)
        overlap: Số conversation trùng giữa 2 window (e.g. 5)
        min_window_size: Window ít nhất bao nhiêu conv mới lấy

    Returns:
        List of lists of Conversation: [window1, window2, ...]
    """
    if len(conversations) == 0:
        return []
    if window_size >= len(conversations):
        return [list(conversations)]

    step = max(1, window_size - overlap)
    windows = []
    start = 0
    while start < len(conversations):
        end = min(start + window_size, len(conversations))
        if end - start >= min_window_size:
            windows.append(conversations[start:end])
        if end >= len(conversations):
            break
        start += step
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
    model: nn.Module,
    users: List[UserData],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    window_size: int,
    window_overlap: int,
    window_aggregation: str,
    max_grad_norm: float = 0.0,
    world_size: int = 1,
) -> Tuple[float, Dict]:
    """Train one epoch. model có thể là DDP-wrapped; users là partition cho rank hiện tại."""
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_samples = 0
    # DDP đã all-reduce và chia trung bình gradient; không cần scale loss.

    for user_data in users:
        if user_data.total_interactions == 0:
            continue

        conversations = user_data.get_conversations_sorted()
        label = user_data.label

        # Positive: split input conversations bằng sliding window → nhiều chuỗi nhỏ → mỗi chuỗi = 1 sample (label=1)
        # Negative: 1 sample = toàn bộ chuỗi conversations
        if label == 1:
            conv_windows = sliding_windows_conversations(
                conversations, window_size, overlap=window_overlap, min_window_size=1
            )
        else:
            conv_windows = [conversations]

        sample_embeddings = []
        for conv_list in conv_windows:
            if not conv_list:
                continue
            # UserData tạm với chuỗi conversations này (cùng target_user, label)
            window_user_data = UserData(
                user_id=user_data.user_id,
                user_id_str=user_data.user_id_str,
                conversations=conv_list,
                label=label,
            )
            if window_user_data.total_interactions == 0:
                continue
            raw_model.reset_state()
            embeddings = model.forward(window_user_data, return_per_event=False)
            if len(embeddings) == 0:
                continue
            # Mỗi chuỗi nhỏ cho 1 embedding (lấy sau conv cuối = last)
            h = aggregate_window(torch.stack(embeddings), window_aggregation)
            sample_embeddings.append(h)

        if len(sample_embeddings) == 0:
            continue

        optimizer.zero_grad()
        batch_agg = torch.stack(sample_embeddings)
        logits = model.classifier(batch_agg)
        labels_t = torch.full(
            (len(sample_embeddings),), label, dtype=torch.long, device=device
        )
        loss = criterion(logits, labels_t)
        loss.backward()
        total_loss += loss.item()
        n_samples += len(sample_embeddings)

        with torch.inference_mode():
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.append(probs.astype(np.float32))
        all_labels.append(np.full(len(sample_embeddings), label, dtype=np.int64))

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    if world_size > 1 and dist.is_initialized():
        total_loss_t = torch.tensor([total_loss, float(n_samples)], device=device, dtype=torch.float64)
        dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
        total_loss = total_loss_t[0].item()
        n_samples = int(total_loss_t[1].item())

    avg_loss = total_loss / max(n_samples, 1)
    metrics = {}
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        if world_size > 1 and dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, (all_preds, all_labels))
            pred_parts = [g[0] for g in gathered if g is not None and len(g[0]) > 0]
            label_parts = [g[1] for g in gathered if g is not None and len(g[1]) > 0]
            all_preds = np.concatenate(pred_parts, axis=0) if pred_parts else np.array([], dtype=np.float32)
            all_labels = np.concatenate(label_parts, axis=0) if label_parts else np.array([], dtype=np.int64)
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
                all_preds.append(0.0)
                all_labels.append(user_data.label)
                continue

            model.reset_state()
            embeddings = model.forward(user_data, return_per_event=False)

            if len(embeddings) == 0:
                all_preds.append(0.0)
                all_labels.append(user_data.label)
                continue

            # Eval: 1 prediction per user (aggregate full sequence)
            h = aggregate_window(torch.stack(embeddings), window_aggregation)
            logits = model.classifier(h.unsqueeze(0))

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
    rank, world_size, local_rank = get_distributed_info()
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        # Chỉ rank 0 ghi log file; mọi rank in ra console
        log_handlers = [logging.StreamHandler()]
        if rank == 0:
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
            log_handlers.append(logging.FileHandler(f"{args.log_dir}/train_{int(time.time())}.log"))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - [rank %s] - %(message)s" % rank,
            handlers=log_handlers,
            force=True,
        )
    else:
        logging.basicConfig(force=True)

    logger = setup_logging(args.log_dir) if not is_distributed else logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    set_seed(args.seed + rank)
    if device is None:
        device = get_device(local_rank if is_distributed else args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {device} (rank {rank}/{world_size})")
    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    use_wandb = getattr(args, "use_wandb", False) and rank == 0
    if use_wandb:
        wandb.init(
            project=getattr(args, "wandb_project", "tgn-erisk"),
            name=getattr(args, "wandb_run_name", None),
            config=vars(args),
            dir=args.log_dir,
        )

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
            verbose=(rank == 0),
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
    train_dataset.print_statistics(verbose=(rank == 0))

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

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_users = train_dataset.users
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
        my_train_users = partition_users_by_rank(train_users, rank, world_size, seed=args.seed + epoch)
        train_loss, train_metrics = train_epoch(
            model=model,
            users=my_train_users,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            window_aggregation=args.window_aggregation,
            max_grad_norm=getattr(args, "max_grad_norm", 1.0),
            world_size=world_size,
        )
        if is_distributed:
            dist.barrier()
        eval_model = model.module if is_distributed else model
        if rank == 0:
            val_loss, val_metrics, _, _ = evaluate(
                model=eval_model,
                dataset=val_dataset,
                device=device,
                window_size=args.window_size,
                window_aggregation=args.window_aggregation,
            )
        else:
            val_loss, val_metrics = 0.0, {}
        if is_distributed:
            val_auc_t = torch.tensor([val_metrics.get("auc", 0.0) if rank == 0 else 0.0], device=device)
            dist.broadcast(val_auc_t, src=0)
            val_auc = val_auc_t.item()
            if rank != 0:
                val_metrics = {"auc": val_auc}
        else:
            val_auc = val_metrics.get("auc", 0.0)
        epoch_time = time.time() - epoch_start

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

            if use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/auc": val_auc,
                    "epoch_time_sec": epoch_time,
                }
                for k, v in train_metrics.items():
                    log_dict[f"train/{k}"] = v
                for k, v in val_metrics.items():
                    log_dict[f"val/{k}"] = v
                wandb.log(log_dict)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            if rank == 0:
                torch.save(eval_model.state_dict(), f"{args.save_dir}/best_model.pth")
                logger.info(f"  New best model saved! (AUC: {val_auc:.4f})")

        if early_stopper.early_stop_check(val_auc):
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

    if rank == 0:
        best_path = f"{args.save_dir}/best_model.pth"
        if not Path(best_path).exists():
            torch.save(eval_model.state_dict(), best_path)
        logger.info(f"Best model: {best_path} (epoch {best_epoch}, val AUC: {best_val_auc:.4f})")
        results = {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "args": vars(args),
        }
        with open(f"{args.save_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.save_dir}/results.json")
        if use_wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_auc"] = best_val_auc
            wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


def main():
    # Chạy 6 GPU: torchrun --nproc_per_node=6 -m tgn_depression.train --data_dir ... (hoặc python -m torch.distributed.run ...)
    parser = argparse.ArgumentParser(description="Train TGNUserSequence for eRisk (single-GPU hoặc multi-GPU DDP)")

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
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Sliding window size (chỉ cho positive user). Số conversation mỗi window.",
    )
    parser.add_argument(
        "--window_overlap",
        type=int,
        default=5,
        help="Overlap giữa 2 sliding windows (chỉ positive user). step = window_size - overlap.",
    )
    parser.add_argument("--window_aggregation", type=str, default="last", choices=["last", "mean"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id when chạy single-GPU")
    parser.add_argument("--max_ego_hops", type=int, default=2)

    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="tgn-erisk", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")

    args = parser.parse_args()
    main_worker(args)


if __name__ == "__main__":
    main()

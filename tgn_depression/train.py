"""
Training script for TGNUserSequence (eRisk).

Sliding-window (chỉ positive user): split chuỗi conversations thành nhiều chuỗi nhỏ
→ mỗi chuỗi nhỏ feed vào model → 1 embedding → 1 sample (label=1). Augment data.
"""

import argparse
import gc
import json
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_user_sequence import TGNUserSequence
from utils.data_structures import DepressionDataset, UserData
from utils.data_loader import (
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, compute_class_weights

# CUDA_VISIBLE_DEVICES nên set từ bên ngoài (torchrun / job) thay vì hardcode
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"


def _log_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """
    Kiểm tra luồng gradient: tính grad norm theo từng module (memory_updater, message_function, ...).
    Trả về dict { "module_name": grad_norm }. Grad None hoặc toàn 0 → coi như 0.
    """
    stats = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            stats[name] = float("nan")
            continue
        norm = g.float().norm().item()
        stats[name] = norm
    # Gộp theo module (prefix)
    by_module = {}
    for name, norm in stats.items():
        parts = name.split(".")
        module = parts[0] if parts else "other"
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(norm)
    out = {}
    for module, norms in by_module.items():
        valid = [n for n in norms if isinstance(n, float) and n == n and n > 0]  # exclude nan, 0
        if valid:
            out[f"grad_norm/{module}"] = (sum(n ** 2 for n in valid) ** 0.5)
        else:
            out[f"grad_norm/{module}"] = 0.0
        out[f"grad_norm/{module}_nonzero"] = len(valid)
    return out

def get_distributed_info():
    """Get rank, world_size, local_rank from env (set by torchrun)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


class UserDataset(Dataset):
    """Dataset bọc list UserData để dùng với DataLoader."""

    def __init__(self, users: List[UserData]):
        self.users = users

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> UserData:
        return self.users[idx]


def _collate_single_user(batch: List) -> UserData:
    """batch_size=1 → trả về đúng 1 UserData."""
    return batch[0]


def build_flat_window_samples(
    users: List[UserData],
    window_size: int,
    window_overlap: int,
) -> List[Tuple[UserData, int]]:
    """
    Flatten users → list of (window_user_data, label). Mỗi window = 1 sample.
    Dùng để DistributedSampler chia samples (không phải users) → cân bằng tải DDP.
    """
    samples: List[Tuple[UserData, int]] = []
    for user_data in users:
        if user_data.total_interactions == 0:
            continue
        conversations = user_data.get_conversations_sorted()
        label = user_data.label
        if label == 1:
            conv_windows = sliding_windows_conversations(
                conversations, window_size, overlap=window_overlap, min_window_size=1
            )
        else:
            conv_windows = [conversations]
        for conv_list in conv_windows:
            if not conv_list:
                continue
            window_user_data = UserData(
                user_id=user_data.user_id,
                user_id_str=user_data.user_id_str,
                conversations=conv_list,
                label=label,
            )
            if window_user_data.total_interactions == 0:
                continue
            samples.append((window_user_data, label))
    return samples


class FlatWindowDataset(Dataset):
    """Dataset của (window_user_data, label) — mỗi sample = 1 window độc lập."""

    def __init__(self, samples: List[Tuple[UserData, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[UserData, int]:
        return self.samples[idx]


def _collate_window_sample(batch: List) -> Tuple[UserData, int]:
    """batch_size=1 → trả về (UserData, label)."""
    return batch[0]


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

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    window_size: int,
    window_overlap: int,
    window_aggregation: str,
    max_grad_norm: float = 0.0,
    world_size: int = 1,
    rank: int = 0,
    check_gradients: bool = False,
    accumulation_steps: int = 1,
) -> Tuple[float, Dict]:
    """
    Train one epoch. Dataloader yields (window_user_data, label) — mỗi sample = 1 window.
    Dùng FlatWindowDataset + DistributedSampler → mỗi rank ~ cùng số samples → cân bằng tải DDP.
    accumulation_steps: gộp gradient từ N sample rồi mới step (effective batch = N).
    """
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_samples = 0
    n_batches_done = 0
    accumulation_steps = max(1, accumulation_steps)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (window_user_data, label) in enumerate(dataloader):
        raw_model.reset_state()
        logits = model.forward(
            window_user_data,
            return_per_event=False,
            return_logits=True,
        )
        if logits is None:
            continue

        label_t = torch.full((1,), int(label), dtype=torch.long, device=device)
        loss = criterion(logits, label_t) / accumulation_steps
        loss.backward()

        n_samples += 1
        with torch.inference_mode():
            prob = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
        all_preds.append(prob.astype(np.float32))
        all_labels.append(int(label))

        if (n_samples % accumulation_steps) == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        n_batches_done += 1

        if check_gradients and rank == 0 and n_batches_done == 1:
            grad_stats = _log_gradient_flow(raw_model)
            print("  [gradient check] after backward:", flush=True)
            for k, v in sorted(grad_stats.items()):
                if k.endswith("_nonzero"):
                    print(f"    {k}: {int(v)}", flush=True)
                else:
                    print(f"    {k}: {v:.6f}", flush=True)
        if rank == 0 and (n_batches_done <= 3 or n_batches_done % 500 == 0):
            print(f"  [train] processed {n_batches_done} samples", flush=True)
        # Chỉ clear cache định kỳ để tránh chậm (mỗi 200 step)
        if device.type == "cuda" and (n_batches_done % 200 == 0):
            torch.cuda.empty_cache()
        if n_batches_done % 500 == 0:
            gc.collect()

    # Flush gradient còn lại khi dùng accumulation
    if n_samples > 0 and (n_samples % accumulation_steps) != 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if world_size > 1 and dist.is_initialized():
        total_loss_t = torch.tensor([total_loss, float(n_samples)], device=device, dtype=torch.float64)
        dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
        total_loss = total_loss_t[0].item()
        n_samples = int(total_loss_t[1].item())

    avg_loss = total_loss / max(n_samples, 1)
    metrics = {}
    if all_preds:
        all_preds = np.array(all_preds, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
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


def compute_metrics_from_preds(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> Dict:
    """Compute classification metrics from prediction probabilities and labels."""
    metrics = {}
    if len(all_labels) == 0:
        return metrics
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
    return metrics


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
            logits = model.forward(
                user_data,
                return_per_event=False,
                return_logits=True,
            )

            if logits is None:
                all_preds.append(0.0)
                all_labels.append(user_data.label)
                continue

            label = user_data.label
            total_loss += criterion(logits, torch.tensor([label], dtype=torch.long, device=device)).item()
            all_preds.append(torch.softmax(logits, dim=1)[0, 1].item())
            all_labels.append(label)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    avg_loss = total_loss / max(len(all_labels), 1)
    metrics = compute_metrics_from_preds(all_preds, all_labels)
    return avg_loss, metrics, all_preds, all_labels


def evaluate_distributed(
    model: TGNUserSequence,
    val_dataset: DepressionDataset,
    device: torch.device,
    window_size: int,
    window_aggregation: str,
    rank: int,
    world_size: int,
) -> Tuple[float, Dict]:
    """
    Distributed eval: each rank evaluates a subset of val users, then we
    all_gather preds/labels and compute global metrics. Avoids timeout and
    reduces eval time by ~world_size.
    """
    # Split val users by rank: rank i gets indices i, i+world_size, i+2*world_size, ...
    my_indices = list(range(rank, len(val_dataset.users), world_size))
    if not my_indices:
        my_preds = np.array([], dtype=np.float32)
        my_labels = np.array([], dtype=np.int64)
        loss_sum = 0.0
        n_samples = 0
    else:
        my_users = [val_dataset.users[i] for i in my_indices]
        val_subset = DepressionDataset(
            users=my_users,
            post_embeddings=val_dataset.post_embeddings,
            n_total_users=val_dataset.n_total_users,
            user_to_idx=val_dataset.user_to_idx,
            idx_to_user=val_dataset.idx_to_user,
        )
        val_loss, _, my_preds, my_labels = evaluate(
            model=model,
            dataset=val_subset,
            device=device,
            window_size=window_size,
            window_aggregation=window_aggregation,
        )
        n_samples = len(my_labels)
        loss_sum = val_loss * max(n_samples, 1)

    # Gather (preds, labels, loss_sum, n_samples) from all ranks
    to_gather = (my_preds, my_labels, float(loss_sum), int(n_samples))
    gathered = [None] * world_size
    dist.all_gather_object(gathered, to_gather)

    # Global loss and metrics
    all_preds = np.concatenate([g[0] for g in gathered], axis=0)
    all_labels = np.concatenate([g[1] for g in gathered], axis=0)
    total_loss_sum = sum(g[2] for g in gathered)
    total_n = sum(g[3] for g in gathered)
    val_loss = total_loss_sum / max(total_n, 1)
    metrics = compute_metrics_from_preds(all_preds, all_labels)
    return val_loss, metrics


def main_worker(args, device=None):
    rank, world_size, local_rank = get_distributed_info()
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")

    # print(
    #     f"PID={os.getpid()} | RANK={rank} | LOCAL_RANK={local_rank} | "
    #     f"CURRENT_DEVICE={torch.cuda.current_device()}"
    # )

    prefix = f"[rank {rank}] " if is_distributed else ""
    print(f"{prefix}Arguments: {args}")

    set_seed(args.seed + rank)
    if device is None:
        device = get_device(local_rank if is_distributed else args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"{prefix}Using device: {device} (rank {rank}/{world_size})")
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

    print(f"{prefix}Loading data...")
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

    print(f"{prefix}Train: {len(train_dataset.users)} users")
    train_dataset.print_statistics(verbose=(rank == 0))

    train_labels = np.array([u.label for u in train_dataset.users])
    class_weights = compute_class_weights(train_labels).to(device).float()

    print(f"{prefix}Initializing TGNUserSequence...")
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

    print(f"{prefix}Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Flatten users → samples (mỗi window = 1 sample) để DistributedSampler cân bằng tải.
    flat_samples = build_flat_window_samples(
        train_dataset.users,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
    )
    flat_dataset = FlatWindowDataset(flat_samples)
    if rank == 0:
        print(f"{prefix}Flat samples: {len(flat_samples)} (từ {len(train_dataset.users)} users)", flush=True)
    if is_distributed:
        train_sampler = DistributedSampler(
            flat_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            flat_dataset,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=_collate_window_sample,
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(getattr(args, "num_workers", 0) > 0),
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            flat_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=_collate_window_sample,
            num_workers=getattr(args, "num_workers", 0),
            persistent_workers=(getattr(args, "num_workers", 0) > 0),
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)

    best_val_auc = 0.0
    best_epoch = 0
    n_batches = len(train_loader)
    print(f"{prefix}Train loader: {n_batches} users per epoch", flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} starting...", flush=True)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            window_aggregation=args.window_aggregation,
            max_grad_norm=getattr(args, "max_grad_norm", 1.0),
            world_size=world_size,
            rank=rank,
            check_gradients=getattr(args, "check_gradients", False),
            accumulation_steps=getattr(args, "accumulation_steps", 1),
        )
        if is_distributed:
            dist.barrier()
        eval_model = model.module if is_distributed else model
        if is_distributed:
            # Distributed eval: each rank evaluates its slice of val set, then all_gather + global metrics.
            val_loss, val_metrics = evaluate_distributed(
                model=eval_model,
                val_dataset=val_dataset,
                device=device,
                window_size=args.window_size,
                window_aggregation=args.window_aggregation,
                rank=rank,
                world_size=world_size,
            )
        else:
            val_loss, val_metrics, _, _ = evaluate(
                model=eval_model,
                dataset=val_dataset,
                device=device,
                window_size=args.window_size,
                window_aggregation=args.window_aggregation,
            )
        val_auc = val_metrics.get("auc", 0.0)
        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
            print(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

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
                print(f"  New best model saved! (AUC: {val_auc:.4f})")

        if early_stopper.early_stop_check(val_auc):
            print(f"Early stopping after {epoch + 1} epochs")
            break

    if rank == 0:
        best_path = f"{args.save_dir}/best_model.pth"
        if not Path(best_path).exists():
            torch.save(eval_model.state_dict(), best_path)
        print(f"Best model: {best_path} (epoch {best_epoch}, val AUC: {best_val_auc:.4f})")
        results = {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "args": vars(args),
        }
        with open(f"{args.save_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_dir}/results.json")
        if use_wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_auc"] = best_val_auc
            wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


def main():
    # Chạy 6 GPU: torchrun --nproc_per_node=6 -m tgn_depression.train --data_dir ... (hoặc python -m torch.distributed.run ...)
    parser = argparse.ArgumentParser(description="Train TGNUserSequence for eRisk (single-GPU hoặc multi-GPU DDP)")

    parser.add_argument("--data_dir", type=str, default="/data/ubuntu-gpu-v100-home/ngocnm32/_luongtd11/temp_2/prune/output/versions/3/eRisk2025_mapped_pruned")
    parser.add_argument("--data_format", type=str, default="parquet_folders")
    parser.add_argument("--neg_folder", type=str, default="neg")
    parser.add_argument("--pos_folder", type=str, default="pos")
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--save_dummy", action="store_true")
    parser.add_argument("--n_total_users", type=int, default=100)
    parser.add_argument("--n_target_users", type=int, default=50)
    parser.add_argument("--n_conversations", type=int, default=200)
    parser.add_argument("--avg_interactions", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--split_method", type=str, default="stratified")

    parser.add_argument("--memory_dim", type=int, default=172)
    parser.add_argument("--n_ego_layers", type=int, default=2)
    parser.add_argument(
        "--embedding_module_type",
        type=str,
        default="graph_attention",
        choices=["identity", "graph_attention", "graph_sum"],
        help="TGN embedding: identity (memory only), graph_attention (paper best), graph_sum",
    )
    parser.add_argument("--n_heads", type=int, default=2, help="Attention heads for graph_attention")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Temporal neighbors per layer (TGN paper: 10)")
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
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation (effective batch = accumulation_steps); tăng tốc khi batch_size=1.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers (0 = main process only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id when chạy single-GPU")
    parser.add_argument("--max_ego_hops", type=int, default=2)

    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="tgn-erisk", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--check_gradients", action="store_true", help="Log gradient norms (memory, classifier, ...) sau backward đầu tiên của epoch 1")

    args = parser.parse_args()
    main_worker(args)


if __name__ == "__main__":
    main()
"""
Training script for TGN Sequential models (CarryOver and LSTM).

Usage:
    # Dummy data
    python train_sequential.py --sequence_mode carryover --use_dummy_data

    # Parquet folders (neg/ + pos/, tên file = target_user)
    python train_sequential.py --data_format parquet_folders --data_dir /path/to/parent_of_neg_pos
    # Hoặc dùng script gọn: python run_train_parquet.py /path/to/parent_of_neg_pos

    # CSV + JSON
    python train_sequential.py --data_dir ./data --data_format csv_json
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_sequential import TGNSequential
from utils.data_structures import UserData, DepressionDataset
from utils.data_loader import (
    load_depression_data,
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, get_device_for_rank, compute_class_weights


class IndexDataset(Dataset):
    """Dataset of indices for DistributedSampler (split users across GPUs)."""
    def __init__(self, n: int):
        self.n = n
    def __len__(self) -> int:
        return self.n
    def __getitem__(self, idx: int) -> int:
        return idx


def setup_logging(log_dir: str = "logs", rank: int = 0):
    """Setup logging. When rank > 0 (DDP), only StreamHandler to avoid multiple processes writing same file."""
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(f'{log_dir}/train_seq_{int(time.time())}.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def train_epoch(model: TGNSequential,
                dataset: DepressionDataset,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                n_neighbors: int = 10,
                user_indices: Optional[List[int]] = None,
                rank: int = 0,
                world_size: int = 1) -> Tuple[float, Dict]:
    """Train for one epoch. If user_indices is set (DDP), only process those users."""
    model.train()
    base_model = model.module if hasattr(model, 'module') else model
    total_loss = 0
    all_preds = []
    all_labels = []
    n_users_processed = 0
    
    indices = user_indices if user_indices is not None else list(range(len(dataset.users)))
    
    for step, user_idx in enumerate(indices):
        user_data = dataset.users[user_idx]
        base_model.reset_state()
        
        if user_data.total_interactions == 0:
            continue
        
        optimizer.zero_grad()
        label = torch.LongTensor([user_data.label]).to(device)
        logits = model(user_data, n_neighbors)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        base_model.detach_memory()
        
        total_loss += loss.item()
        n_users_processed += 1
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.append(user_data.label)
        
        if rank == 0 and (step + 1) % 50 == 0:
            logging.info(f"  Processed {step + 1}/{len(indices)} users (rank 0)")
    
    avg_loss = total_loss / max(n_users_processed, 1)
    metrics = {}
    if len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_labels = (all_preds > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(all_labels, pred_labels)
        if len(np.unique(all_labels)) > 1:
            metrics['auc'] = roc_auc_score(all_labels, all_preds)
            metrics['f1'] = f1_score(all_labels, pred_labels)
            precision, recall, _, _ = precision_recall_fscore_support(
                all_labels, pred_labels, average='binary', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
    
    return avg_loss, metrics


def evaluate(model: TGNSequential,
             dataset: DepressionDataset,
             device: torch.device,
             n_neighbors: int = 10,
             user_indices: Optional[List[int]] = None) -> Tuple[float, Dict]:
    """Evaluate model. If user_indices is set (DDP), only evaluate on those users."""
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_users_processed = 0
    
    indices = user_indices if user_indices is not None else list(range(len(dataset.users)))
    
    with torch.no_grad():
        for user_idx in indices:
            user_data = dataset.users[user_idx]
            base_model.reset_state()
            if user_data.total_interactions == 0:
                continue
            label = torch.LongTensor([user_data.label]).to(device)
            logits = model(user_data, n_neighbors)
            loss = criterion(logits, label)
            total_loss += loss.item()
            n_users_processed += 1
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.append(user_data.label)
    
    avg_loss = total_loss / max(n_users_processed, 1)
    metrics = {}
    preds_arr = np.array(all_preds) if all_preds else np.array([])
    labels_arr = np.array(all_labels) if all_labels else np.array([])
    if len(all_preds) > 0:
        pred_labels = (preds_arr > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(labels_arr, pred_labels)
        if len(np.unique(labels_arr)) > 1:
            metrics['auc'] = roc_auc_score(labels_arr, preds_arr)
            metrics['f1'] = f1_score(labels_arr, pred_labels)
            precision, recall, _, _ = precision_recall_fscore_support(
                labels_arr, pred_labels, average='binary', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
    return avg_loss, metrics, preds_arr, labels_arr


def main_worker(args,
                device: Optional[torch.device] = None,
                rank: int = 0,
                world_size: int = 1):
    """Main training function. When world_size > 1, runs under DDP (device/rank set by spawn)."""
    logger = setup_logging(args.log_dir, rank=rank)
    if rank == 0:
        logger.info(f"Arguments: {args}")
        logger.info(f"Sequence mode: {args.sequence_mode.upper()}")
    
    set_seed(args.seed + rank)
    if device is None:
        device = get_device(args.gpu)
    if rank == 0:
        logger.info(f"Using device: {device} (rank {rank}/{world_size})")
    
    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    
    # Load data
    logger.info("Loading data...")
    if args.use_dummy_data:
        train_dataset, val_dataset, test_dataset, metadata = create_dummy_data(
            n_total_users=args.n_total_users,
            n_target_users=args.n_target_users,
            n_conversations=args.n_conversations,
            avg_interactions=args.avg_interactions,
            embedding_dim=args.embedding_dim,
            depression_ratio=0.3,
            save_dir=args.data_dir if args.save_dummy else None
        )
    elif getattr(args, "data_format", "csv_json") == "parquet_folders":
        # Chỉ chia train/val, không chia tập test (test chạy riêng với run_test_parquet.py)
        train_dataset, val_dataset, test_dataset, metadata = load_depression_data_from_parquet_folders(
            data_dir=args.data_dir,
            neg_folder=args.neg_folder,
            pos_folder=args.pos_folder,
            val_ratio=0.15,
            test_ratio=0.0,
            split_method=args.split_method,
            seed=args.seed
        )
    else:
        train_dataset, val_dataset, test_dataset, metadata = load_depression_data(
            interactions_path=f"{args.data_dir}/interactions.csv",
            embeddings_path=f"{args.data_dir}/embeddings.json",
            labels_path=f"{args.data_dir}/labels.json",
            split_method=args.split_method,
            seed=args.seed
        )
    
    logger.info("Training dataset:")
    train_dataset.print_statistics()
    
    # Compute class weights
    train_labels = np.array([u.label for u in train_dataset.users])
    class_weights = compute_class_weights(train_labels).to(device).float()
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize model
    logger.info(f"Initializing TGN Sequential model ({args.sequence_mode} mode)...")
    model = TGNSequential(
        n_users=metadata['n_total_users'],
        edge_features=train_dataset.post_embeddings,
        device=device,
        # Sequence mode
        sequence_mode=args.sequence_mode,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_bidirectional=args.lstm_bidirectional,
        # TGN params
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        use_memory=args.use_memory,
        memory_dimension=args.memory_dim,
        embedding_module_type=args.embedding_module,
        message_function_type=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=args.n_neighbors,
        num_classes=2
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[0], find_unused_parameters=True)
    
    if rank == 0:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
    
    train_index_ds = IndexDataset(len(train_dataset.users))
    val_index_ds = IndexDataset(len(val_dataset.users))
    train_sampler = DistributedSampler(train_index_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_index_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    train_indices = list(train_sampler) if train_sampler is not None else None
    val_indices = list(val_sampler) if val_sampler is not None else None
    
    if rank == 0:
        logger.info("Starting training...")
    best_val_auc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            train_indices = list(train_sampler)
        
        epoch_start = time.time()
        
        train_loss, train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_neighbors=args.n_neighbors,
            user_indices=train_indices,
            rank=rank,
            world_size=world_size
        )
        
        val_loss, val_metrics, val_preds, val_labels = evaluate(
            model=model,
            dataset=val_dataset,
            device=device,
            n_neighbors=args.n_neighbors,
            user_indices=val_indices
        )
        
        if world_size > 1:
            preds_t = torch.from_numpy(val_preds.astype(np.float32)).to(device)
            labels_t = torch.from_numpy(val_labels.astype(np.int64)).to(device)
            local_size = torch.tensor([preds_t.shape[0]], dtype=torch.long).to(device)
            size_list = [torch.zeros(1, dtype=torch.long).to(device) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            max_size = max(s.item() for s in size_list)
            if max_size == 0:
                val_metrics = {}
                val_auc = 0.0
            else:
                if preds_t.shape[0] == 0:
                    preds_t = torch.zeros(max_size, dtype=torch.float32).to(device)
                    labels_t = torch.full((max_size,), -1, dtype=torch.int64).to(device)
                elif preds_t.shape[0] < max_size:
                    preds_t = torch.nn.functional.pad(preds_t, (0, max_size - preds_t.shape[0]), value=0)
                    labels_t = torch.nn.functional.pad(labels_t, (0, max_size - labels_t.shape[0]), value=-1)
                preds_list = [torch.zeros_like(preds_t).to(device) for _ in range(world_size)]
                labels_list = [torch.zeros_like(labels_t).to(device) for _ in range(world_size)]
                dist.all_gather(preds_list, preds_t)
                dist.all_gather(labels_list, labels_t)
                if rank == 0:
                    all_preds = torch.cat([p[:s.item()] for p, s in zip(preds_list, size_list)], dim=0).cpu().numpy()
                    all_labels = torch.cat([l[:s.item()] for l, s in zip(labels_list, size_list)], dim=0).cpu().numpy()
                    valid = all_labels >= 0
                    all_preds, all_labels = all_preds[valid], all_labels[valid]
                    if len(all_preds) > 0 and len(np.unique(all_labels)) > 1:
                        val_metrics = {
                            'accuracy': accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
                            'auc': roc_auc_score(all_labels, all_preds),
                            'f1': f1_score(all_labels, (all_preds > 0.5).astype(int))
                        }
                        pr, rc, _, _ = precision_recall_fscore_support(all_labels, (all_preds > 0.5).astype(int), average='binary', zero_division=0)
                        val_metrics['precision'], val_metrics['recall'] = pr, rc
                val_auc_local = val_metrics.get('auc', 0.0)
                val_auc_t = torch.tensor([val_auc_local], dtype=torch.float32).to(device)
                dist.broadcast(val_auc_t, src=0)
                val_auc = val_auc_t.item()
        else:
            val_auc = val_metrics.get('auc', 0)
        
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            if rank == 0:
                save_model = model.module if hasattr(model, 'module') else model
                torch.save(save_model.state_dict(), f"{args.save_dir}/best_model_{args.sequence_mode}.pth")
                logger.info(f"  New best model saved! (AUC: {val_auc:.4f})")
        
        if early_stopper.early_stop_check(val_auc):
            if rank == 0:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    best_path = f"{args.save_dir}/best_model_{args.sequence_mode}.pth"
    if rank == 0:
        if not Path(best_path).exists():
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), best_path)
            logger.info("No best model saved (val AUC did not improve); saved last model.")
        logger.info(f"\nBest model saved: {best_path} (epoch {best_epoch}, val AUC: {best_val_auc:.4f})")
        
        import json
        results = {
            'sequence_mode': args.sequence_mode,
            'best_epoch': best_epoch,
            'best_val_auc': best_val_auc,
            'args': vars(args)
        }
        with open(f"{args.save_dir}/results_{args.sequence_mode}.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.save_dir}/results_{args.sequence_mode}.json")
        logger.info("Chạy test riêng: python run_test_parquet.py <data_dir> <model_path>")
    
    if world_size > 1:
        dist.destroy_process_group()


def worker_fn(rank: int, args, n_gpus: int, gpu_ids: List[int]):
    """Entry point for each GPU process (DDP)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Set device before init_process_group (after CUDA_VISIBLE_DEVICES, cuda:0 is the only visible GPU)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank, world_size=n_gpus)
    dist.barrier()  # Ensure all ranks are ready before proceeding
    main_worker(args, device=device, rank=rank, world_size=n_gpus)


def main(args):
    """Entry point. Single-GPU or multi-GPU via spawn."""
    if getattr(args, 'multi_gpu', False):
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        n_gpus = len(gpu_ids)
        mp.spawn(worker_fn, args=(args, n_gpus, gpu_ids), nprocs=n_gpus, join=True)
    else:
        main_worker(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TGN Sequential for Depression Detection')
    
    # Sequence mode
    parser.add_argument('--sequence_mode', type=str, default='carryover',
                        choices=['carryover', 'lstm'],
                        help='Sequence processing mode: carryover or lstm')
    
    # LSTM parameters
    parser.add_argument('--lstm_hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (for lstm mode)')
    parser.add_argument('--lstm_num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm_bidirectional', action='store_true',
                        help='Use bidirectional LSTM')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data files (or parent of neg/pos for parquet_folders)')
    parser.add_argument('--data_format', type=str, default='csv_json',
                        choices=['csv_json', 'parquet_folders'],
                        help='Data format: csv_json (interactions.csv + embeddings.json + labels.json) or parquet_folders (neg/ + pos/)')
    parser.add_argument('--neg_folder', type=str, default='neg',
                        help='Folder name for label 0 (parquet_folders only)')
    parser.add_argument('--pos_folder', type=str, default='pos',
                        help='Folder name for label 1 (parquet_folders only)')
    parser.add_argument('--use_dummy_data', action='store_true',
                        help='Use generated dummy data for testing')
    parser.add_argument('--save_dummy', action='store_true',
                        help='Save generated dummy data')
    parser.add_argument('--n_total_users', type=int, default=100,
                        help='Total number of users for dummy data')
    parser.add_argument('--n_target_users', type=int, default=50,
                        help='Number of target users for dummy data')
    parser.add_argument('--n_conversations', type=int, default=200,
                        help='Number of conversations for dummy data')
    parser.add_argument('--avg_interactions', type=int, default=10,
                        help='Average interactions per conversation')
    parser.add_argument('--embedding_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--split_method', type=str, default='stratified',
                        choices=['stratified', 'random'],
                        help='Train/val/test split: stratified or random')
    
    # TGN Model arguments
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of graph attention layers')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--use_memory', action='store_true', default=True,
                        help='Use memory module')
    parser.add_argument('--no_memory', action='store_false', dest='use_memory',
                        help='Disable memory module')
    parser.add_argument('--memory_dim', type=int, default=172,
                        help='Memory dimension')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of neighbors to sample')
    parser.add_argument('--embedding_module', type=str, default='graph_attention',
                        choices=['graph_attention', 'graph_sum', 'identity'],
                        help='Embedding module type')
    parser.add_argument('--message_function', type=str, default='identity',
                        choices=['identity', 'mlp'],
                        help='Message function type')
    parser.add_argument('--aggregator', type=str, default='last',
                        choices=['last', 'mean'],
                        help='Message aggregator type')
    parser.add_argument('--memory_updater', type=str, default='gru',
                        choices=['gru', 'rnn'],
                        help='Memory updater type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (single-GPU mode)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Train on multiple GPUs (DDP)')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                        help='Comma-separated GPU IDs for multi-GPU (e.g. 0,1 for 2x T4)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    main(args)

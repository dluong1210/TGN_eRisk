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
import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score

from model.tgn_sequential import TGNSequential
from utils.data_structures import UserData, DepressionDataset
from utils.data_loader import (
    load_depression_data,
    load_depression_data_from_parquet_folders,
    create_dummy_data,
)
from utils.utils import EarlyStopMonitor, set_seed, get_device, compute_class_weights


def slice_positive_users(
    dataset: DepressionDataset,
    window_size: int,
    overlap: int,
) -> DepressionDataset:
    """
    Với user có label=1 (dương tính), slice chuỗi conversations thành nhiều sample
    theo sliding window: window_size conversations/sample, overlap conversations giữa 2 window.
    Ví dụ: window_size=100, overlap=10 → [0:100], [90:190], [180:280], ...
    Xử lý:
    - len(conversations) < window_size: giữ 1 sample với toàn bộ conversations (không slice).
    - Sample cuối khi slice có thể ít hơn window_size (partial window) — vẫn giữ.
    - overlap phải < window_size; nếu không thì dùng overlap = window_size - 1.
    """
    if window_size < 1:
        return dataset
    step = window_size - overlap
    if step < 1:
        overlap = max(0, window_size - 1)
        step = 1

    new_users: List[UserData] = []
    for u in dataset.users:
        if u.label == 0:
            new_users.append(u)
            continue
        convs = u.get_conversations_sorted()
        n = len(convs)
        if n < window_size:
            # Quá ít conversation: giữ nguyên 1 sample (không slice)
            new_users.append(u)
            continue
        # Sliding window: [0, step, 2*step, ...]; mỗi window [start : start+window_size], cho phép window cuối ngắn hơn
        start = 0
        while start < n:
            end = min(start + window_size, n)
            window_convs = convs[start:end]
            if len(window_convs) == 0:
                break
            new_users.append(
                UserData(
                    user_id=u.user_id,
                    user_id_str=u.user_id_str,
                    conversations=window_convs,
                    label=1,
                )
            )
            if end >= n:
                break
            start += step

    return DepressionDataset(
        users=new_users,
        post_embeddings=dataset.post_embeddings,
        n_total_users=dataset.n_total_users,
        user_to_idx=dataset.user_to_idx,
        idx_to_user=dataset.idx_to_user,
    )


def setup_logging(log_dir: str = "logs"):
    """Setup logging to console and file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f'{log_dir}/train_seq_{int(time.time())}.log'),
    ]
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
                user_batch_size: int = 1,
                use_batch_forward: bool = False) -> Tuple[float, Dict]:
    """Train for one epoch. use_batch_forward=True (LSTM only): 1 forward cho cả batch user (merged graph), nhanh hơn nhiều."""
    model.train()
    base_model = model.module if hasattr(model, 'module') else model
    total_loss = 0
    all_preds = []
    all_labels = []
    n_users_processed = 0
    use_merged = use_batch_forward and getattr(base_model, 'sequence_mode', None) == 'lstm'

    indices = list(range(len(dataset.users)))
    user_batch_size = max(1, int(user_batch_size))

    for batch_start in range(0, len(indices), user_batch_size):
        batch_indices = indices[batch_start:batch_start + user_batch_size]
        if len(batch_indices) == 0:
            continue

        optimizer.zero_grad()
        n_in_batch = len(batch_indices)
        batch_users = [dataset.users[i] for i in batch_indices]
        batch_labels = np.array([u.label for u in batch_users], dtype=np.int64)
        labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

        if use_merged and n_in_batch > 0:
            # Một forward cho cả batch (merged graph) — nhanh hơn rất nhiều so với từng user
            logits = base_model.forward_merged_graph(batch_users, n_neighbors)
            loss = criterion(logits, labels_t)
            loss.backward()
            total_loss += loss.item()
            n_users_processed += n_in_batch
            with torch.inference_mode():
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.append(probs.astype(np.float32))
            all_labels.append(batch_labels)
            del logits, loss
        else:
            # Gradient accumulation: từng user forward/backward (tránh OOM khi không dùng merged)
            batch_probs = []
            batch_labels_list = []
            for user_idx in batch_indices:
                user_data = dataset.users[user_idx]
                base_model.reset_state()
                logits = model(user_data, n_neighbors)
                batch_labels_list.append(user_data.label)
                loss = criterion(logits, torch.tensor([user_data.label], dtype=torch.long, device=device))
                (loss / n_in_batch).backward()
                base_model.detach_memory()
                total_loss += loss.item()
                n_users_processed += 1
                with torch.inference_mode():
                    p = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy().item()
                batch_probs.append(p)
                del logits, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if batch_probs:
                all_preds.append(np.array(batch_probs, dtype=np.float32))
                all_labels.append(np.array(batch_labels_list, dtype=np.int64))

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        optimizer.step()

        if ((batch_start // user_batch_size) + 1) % 100 == 0:
            logging.info(f"  Processed {batch_start + len(batch_indices)}/{len(indices)} users")
    
    avg_loss = total_loss / max(n_users_processed, 1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {}
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
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
             eval_batch_size: int = 1) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    """Evaluate model. eval_batch_size > 1 (LSTM mode) gom nhiều user một forward để eval nhanh hơn."""
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    indices = list(range(len(dataset.users)))
    n_indices = len(indices)
    preds_arr = np.empty(n_indices, dtype=np.float32)
    labels_arr = np.empty(n_indices, dtype=np.int64)
    use_batch_eval = (
        eval_batch_size > 1
        and base_model.sequence_mode == 'lstm'
        and n_indices > 1
    )
    eval_batch_size = min(eval_batch_size, n_indices) if use_batch_eval else 1

    with torch.inference_mode():
        if use_batch_eval:
            for start in range(0, n_indices, eval_batch_size):
                end = min(start + eval_batch_size, n_indices)
                batch_users = [dataset.users[indices[i]] for i in range(start, end)]
                logits = base_model.forward_batch_users(batch_users, n_neighbors)
                for i in range(end - start):
                    pos = start + i
                    label = dataset.users[indices[pos]].label
                    labels_arr[pos] = label
                    total_loss += criterion(logits[i : i + 1], torch.tensor([label], dtype=torch.long, device=device)).item()
                    preds_arr[pos] = torch.softmax(logits[i : i + 1], dim=1)[0, 1].item()
        else:
            for i, user_idx in enumerate(indices):
                user_data = dataset.users[user_idx]
                base_model.reset_state()
                label = user_data.label
                labels_arr[i] = label
                label_t = torch.tensor([label], dtype=torch.long, device=device)
                logits = model(user_data, n_neighbors)
                total_loss += criterion(logits, label_t).item()
                preds_arr[i] = torch.softmax(logits, dim=1)[0, 1].item()

    avg_loss = total_loss / max(n_indices, 1)
    metrics = {}
    if n_indices > 0:
        pred_labels = (preds_arr > 0.5).astype(np.int32)
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


def main_worker(args, device: Optional[torch.device] = None):
    """Main training function (single GPU)."""
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Sequence mode: {args.sequence_mode.upper()}")

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
            save_dir=args.data_dir if args.save_dummy else None
        )
    elif getattr(args, "data_format", "csv_json") == "parquet_folders":
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
    # else:
    #     train_dataset, val_dataset, test_dataset, metadata = load_depression_data(
    #         interactions_path=f"{args.data_dir}/interactions.csv",
    #         embeddings_path=f"{args.data_dir}/embeddings.json",
    #         labels_path=f"{args.data_dir}/labels.json",
    #         val_ratio=args.val_ratio,
    #         test_ratio=args.test_ratio,
    #         split_method=args.split_method,
    #         seed=args.seed
    #     )
    
    def _one_line_stats(ds: DepressionDataset) -> str:
        n = len(ds.users)
        if n == 0:
            return "0 users"
        n_pos = sum(1 for u in ds.users if u.label == 1)
        return f"{n} users ({n_pos} pos, {n - n_pos} neg)"

    def _drop_empty_conversations(ds: DepressionDataset) -> int:
        """Bỏ các conversation không có event nào (n_interactions==0), không đưa vào model. Trả về số conv đã bỏ."""
        dropped = 0
        for u in ds.users:
            before = len(u.conversations)
            u.conversations = [c for c in u.conversations if c.n_interactions > 0]
            dropped += before - len(u.conversations)
        return dropped

    logger.info("Training dataset: %s", _one_line_stats(train_dataset))

    # Chỉ drop empty conversations ở train/val; giữ nguyên tập test (chạy test đúng như run_test_parquet)
    total_dropped = _drop_empty_conversations(train_dataset) + _drop_empty_conversations(val_dataset)
    if total_dropped > 0:
        logger.info("Dropped %d empty conversations (no events) in train/val.", total_dropped)

    def _filter_nonempty(ds: DepressionDataset) -> DepressionDataset:
        before = len(ds.users)
        ds.users = [u for u in ds.users if u.total_interactions > 0]
        removed = before - len(ds.users)
        if removed > 0:
            logger.info("Filtered out %d empty users (no interactions); %d remaining.", removed, len(ds.users))
        return ds

    train_dataset = _filter_nonempty(train_dataset)
    val_dataset = _filter_nonempty(val_dataset)
    # Không filter test: tập test giữ nguyên để đánh giá đúng như run_test_parquet (có thể có user/conv rỗng, script test đã xử lý skip trong run)

    # Slice positive users theo window_size / overlap (chỉ train): mỗi user dương tính thành nhiều sample (sliding window) → số sample tăng, tỷ lệ pos tăng
    if getattr(args, 'positive_window_size', None) and args.positive_window_size >= 1:
        overlap = getattr(args, 'positive_overlap', 0) or 0
        logger.info("Slicing positive users: window_size=%s, overlap=%s", args.positive_window_size, overlap)
        train_dataset = slice_positive_users(train_dataset, args.positive_window_size, overlap)
        logger.info("After slicing: %s (mỗi user dương tính → nhiều window-samples)", _one_line_stats(train_dataset))

    logger.info("Final training set:")
    train_dataset.print_statistics(verbose=True)

    train_pos = np.mean([u.label for u in train_dataset.users])
    val_pos = np.mean([u.label for u in val_dataset.users])
    logger.info(f"Train pos ratio: {train_pos:.4f}, Val pos ratio: {val_pos:.4f} (baseline predict-all-1 F1≈{2*val_pos/(1+val_pos):.4f})")

    train_labels = np.array([u.label for u in train_dataset.users])
    class_weights = compute_class_weights(train_labels).to(device).float()
    logger.info(f"Class weights: {class_weights}")

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
        num_classes=2,
        conversation_batch_size=getattr(args, "conversation_batch_size", 500),
        max_conversations_per_user=None if getattr(args, "max_conversations_per_user", 80) == -1 else getattr(args, "max_conversations_per_user", 80),
        use_carryover_grad=getattr(args, "use_carryover_grad", True),
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)

    user_batch_size = getattr(args, "user_batch_size", 4)
    avg_conv = getattr(train_dataset, "avg_conversations_per_user", 0) or 0
    n_total = metadata.get("n_total_users", 0) or getattr(train_dataset, "n_total_users", 0)
    if n_total > 10000 and avg_conv > 50 and user_batch_size > 8:
        suggested = min(user_batch_size, 8)
        logger.info(
            f"Dataset lớn (n_total_users={n_total}, avg_conversations_per_user={avg_conv:.0f}). "
            f"Gợi ý: --user_batch_size {suggested} nếu CPU 100%% hoặc VRAM tăng (hiện tại {user_batch_size})."
        )

    use_batch_forward = getattr(args, "use_batch_forward", False)
    max_conv = getattr(args, "max_conversations_per_user", 80)
    if use_batch_forward and getattr(model.module if hasattr(model, 'module') else model, 'sequence_mode', None) == 'lstm':
        logger.info(
            "use_batch_forward=True: training với merged graph (1 forward/batch) — nhanh hơn nhiều. "
            "Logic tương đương từng user khi participants mỗi target không trùng nhau (xem doc forward_merged_graph)."
        )
    elif use_batch_forward:
        logger.info("use_batch_forward chỉ có hiệu lực với --sequence_mode lstm (đang bỏ qua).")
    if avg_conv > 30 and max_conv not in (-1, None) and max_conv > 30:
        logger.info(
            f"Gợi ý tốc độ: --max_conversations_per_user 20 hoặc 30 (hiện tại {max_conv}) "
            "để train nhanh hơn nếu vẫn chậm."
        )
    logger.info("Starting training...")
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_start = time.time()
        train_loss, train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_neighbors=args.n_neighbors,
            user_batch_size=user_batch_size,
            use_batch_forward=getattr(args, "use_batch_forward", False),
        )
        train_time = time.time() - train_start

        eval_start = time.time()
        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            dataset=val_dataset,
            device=device,
            n_neighbors=args.n_neighbors,
            eval_batch_size=getattr(args, "eval_batch_size", 8),
        )
        eval_time = time.time() - eval_start

        val_auc = val_metrics.get('auc', 0.0)
        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) train={train_time:.1f}s eval={eval_time:.1f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), f"{args.save_dir}/best_model_{args.sequence_mode}.pth")
            logger.info(f"  New best model saved! (AUC: {val_auc:.4f})")

        if early_stopper.early_stop_check(val_auc):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    best_path = f"{args.save_dir}/best_model_{args.sequence_mode}.pth"
    if not Path(best_path).exists():
        save_model = model.module if hasattr(model, 'module') else model
        torch.save(save_model.state_dict(), best_path)
        logger.info("No best model saved (val AUC did not improve); saved last model.")
    logger.info(f"\nBest model saved: {best_path} (epoch {best_epoch}, val AUC: {best_val_auc:.4f})")

    # Đánh giá test chuẩn eRisk (build_temporal_run + METRICS: ERDE, latency, ...)
    test_metrics = None
    if len(test_dataset.users) > 0:
        logger.info("Evaluating on test set (eRisk standard)...")
        save_model = model.module if hasattr(model, 'module') else model
        save_model.load_state_dict(torch.load(best_path, map_location=device))
        try:
            from run_test_parquet import build_temporal_run
            from utils.metrics import METRICS
            threshold = getattr(args, "threshold", 0.5)
            temporal_run, golden = build_temporal_run(
                model=model,
                dataset=test_dataset,
                device=device,
                n_neighbors=args.n_neighbors,
                threshold=threshold,
            )
            test_metrics = {}
            for name, (metric_fn, _) in METRICS.items():
                test_metrics[name] = float(metric_fn(temporal_run, golden))
            logger.info("Test (eRisk): %s", test_metrics)
        except Exception as e:
            logger.warning("Test eRisk metrics skipped: %s", e)

    results = {
        'sequence_mode': args.sequence_mode,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'args': vars(args)
    }
    if test_metrics is not None:
        results['test_metrics'] = test_metrics
    with open(f"{args.save_dir}/results_{args.sequence_mode}.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.save_dir}/results_{args.sequence_mode}.json")
    if len(test_dataset.users) == 0:
        logger.info("Chạy test riêng: python run_test_parquet.py <data_dir> <model_path>")


def main(args):
    """Entry point (single GPU)."""
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
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Tỉ lệ train (6:1:3 → 0.6). Chỉ dùng khi data_format parquet_folders hoặc csv_json.')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Tỉ lệ validation (6:1:3 → 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Tỉ lệ test (6:1:3 → 0.3). Sau train sẽ tự đánh giá trên test set.')
    parser.add_argument('--split_method', type=str, default='stratified',
                        choices=['stratified', 'random'],
                        help='Train/val/test split: stratified (giữ tỷ lệ label) hoặc random')
    parser.add_argument('--positive_window_size', type=int, default=None,
                        help='Slice user dương tính: số conversation mỗi sample. None/0 = không slice.')
    parser.add_argument('--positive_overlap', type=int, default=10,
                        help='Số conversation trùng nhau giữa 2 window khi slice (chỉ dùng khi positive_window_size set).')
    
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
    parser.add_argument('--conversation_batch_size', type=int, default=200,
                    help='Batch size cho interactions trong mỗi conversation (TGNSequential)')
    parser.add_argument('--max_conversations_per_user', type=int, default=80,
                    help='Chỉ dùng K conversation gần nhất mỗi user để tránh OOM (user 166 conv → cap 80). -1 = không giới hạn.')
    
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
                        help='GPU index')
    parser.add_argument('--user_batch_size', type=int, default=4,
                        help='Số lượng user xử lý trước mỗi lần optimizer.step(). Tăng 8-16 nếu đủ VRAM để train nhanh hơn.')
    parser.add_argument('--use_batch_forward', action='store_true',
                        help='LSTM only: 1 forward merged graph cho cả batch user (nhanh hơn nhiều, nên bật khi data lớn).')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Số user gom một lần khi evaluate (LSTM mode). Tăng để val nhanh hơn.')
    parser.add_argument('--use_carryover_grad', action='store_true', default=True,
                        help='Carryover: bật gradient qua toàn chuỗi (default). --no_use_carryover_grad để tắt (tiết kiệm VRAM).')
    parser.add_argument('--no_use_carryover_grad', action='store_false', dest='use_carryover_grad',
                        help='Carryover: tắt gradient qua chuỗi, chỉ học từ conversation cuối.')
    parser.add_argument('--max_ego_hops', type=int, default=2,
                        help='Khi load parquet: chỉ giữ event trong L-hop ego (0/1/2). -1 = tắt (load full).')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ngưỡng quyết định khi đánh giá test (p>=threshold => positive), dùng cho temporal metrics như run_test_parquet')
    
    args = parser.parse_args()
    main(args)

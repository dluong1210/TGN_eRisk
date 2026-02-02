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
import time
from pathlib import Path
from typing import Dict, Tuple
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


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/train_seq_{int(time.time())}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model: TGNSequential,
                dataset: DepressionDataset,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                n_neighbors: int = 10) -> Tuple[float, Dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_users_processed = 0
    
    for user_idx, user_data in enumerate(dataset.users):
        # Reset state for new user
        model.reset_state()
        
        # Skip users with no interactions
        if user_data.total_interactions == 0:
            continue
        
        # Forward pass
        optimizer.zero_grad()
        
        label = torch.LongTensor([user_data.label]).to(device)
        logits = model.forward_user(user_data, n_neighbors)
        
        # Compute loss
        loss = criterion(logits, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Detach memory
        model.detach_memory()
        
        # Track metrics
        total_loss += loss.item()
        n_users_processed += 1
        
        # Get predictions
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.append(user_data.label)
        
        # Log progress
        if (user_idx + 1) % 50 == 0:
            logging.info(f"  Processed {user_idx + 1}/{len(dataset.users)} users")
    
    # Compute metrics
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
             n_neighbors: int = 10) -> Tuple[float, Dict]:
    """Evaluate model on dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    n_users_processed = 0
    
    with torch.no_grad():
        for user_data in dataset.users:
            model.reset_state()
            
            if user_data.total_interactions == 0:
                continue
            
            label = torch.LongTensor([user_data.label]).to(device)
            logits = model.forward_user(user_data, n_neighbors)
            
            loss = criterion(logits, label)
            total_loss += loss.item()
            n_users_processed += 1
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.append(user_data.label)
    
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


def main(args):
    """Main training function."""
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Sequence mode: {args.sequence_mode.upper()}")
    
    set_seed(args.seed)
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
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
        train_dataset, val_dataset, test_dataset, metadata = load_depression_data_from_parquet_folders(
            data_dir=args.data_dir,
            neg_folder=args.neg_folder,
            pos_folder=args.pos_folder,
            val_ratio=0.15,
            test_ratio=0.15,
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
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
    
    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_neighbors=args.n_neighbors
        )
        
        val_loss, val_metrics = evaluate(
            model=model,
            dataset=val_dataset,
            device=device,
            n_neighbors=args.n_neighbors
        )
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f}, Metrics: {train_metrics}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
        
        val_auc = val_metrics.get('auc', 0)
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{args.save_dir}/best_model_{args.sequence_mode}.pth")
            logger.info(f"  New best model saved! (AUC: {val_auc:.4f})")
        
        if early_stopper.early_stop_check(val_auc):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Test
    best_path = f"{args.save_dir}/best_model_{args.sequence_mode}.pth"
    if not Path(best_path).exists():
        torch.save(model.state_dict(), best_path)
        logger.info(f"No best model saved (val AUC did not improve); using last model for test.")
    logger.info(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_path))
    
    test_loss, test_metrics = evaluate(
        model=model,
        dataset=test_dataset,
        device=device,
        n_neighbors=args.n_neighbors
    )
    
    logger.info("=" * 50)
    logger.info(f"Final Test Results ({args.sequence_mode.upper()} mode):")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Metrics: {test_metrics}")
    logger.info("=" * 50)
    
    # Save results
    import json
    results = {
        'sequence_mode': args.sequence_mode,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    
    with open(f"{args.save_dir}/results_{args.sequence_mode}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.save_dir}/results_{args.sequence_mode}.json")


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
                        help='GPU index')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    main(args)

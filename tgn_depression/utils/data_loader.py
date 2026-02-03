"""
Data loading utilities for TGN Depression Detection.

Data format (option 1 - CSV + JSON):
- CSV file: userID, parentID, timestamp, post_id, conversation_id
- JSON embeddings: {"target_user_id": {"post_id": [embedding_vector]}}
- JSON labels: {"target_user_id": 0 or 1}

Data format (option 2 - Parquet folders):
- data_dir/neg/  (label 0) và data_dir/pos/  (label 1)
- Mỗi folder chứa các file .parquet, tên file = target user id
- Mỗi parquet: userID, parentID, timestamp, post_id, conversation_id, embedding
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from .data_structures import Conversation, UserData, DepressionDataset
except ImportError:
    from data_structures import Conversation, UserData, DepressionDataset


def _embedding_to_1d(emb):
    """Convert embedding (list/array, possibly nested) to 1D np.float32 without np.stack on ragged data."""
    if emb is None or (isinstance(emb, float) and np.isnan(emb)):
        return np.array([], dtype=np.float32)
    flat = []

    def _flatten(x):
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            for y in x:
                _flatten(y)
        else:
            flat.append(float(x))

    _flatten(emb)
    return np.array(flat, dtype=np.float32)


def _stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split: giữ tỉ lệ từng class trong train và phần còn lại.
    
    Args:
        indices: Indices của samples
        labels: Labels tương ứng
        train_ratio: Tỉ lệ cho phần đầu (train)
        rng: Random state
    
    Returns:
        train_indices, rest_indices
    """
    train_idx = []
    rest_idx = []
    
    for label_val in np.unique(labels):
        mask = labels == label_val
        class_indices = indices[mask]
        rng.shuffle(class_indices)
        n_train = max(1, int(len(class_indices) * train_ratio))
        train_idx.extend(class_indices[:n_train])
        rest_idx.extend(class_indices[n_train:])
    
    rng.shuffle(train_idx)
    rng.shuffle(rest_idx)
    return np.array(train_idx), np.array(rest_idx)


def load_depression_data(
    interactions_path: str,
    embeddings_path: str,
    labels_path: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_method: str = "stratified",
    seed: int = 42
) -> Tuple[DepressionDataset, DepressionDataset, DepressionDataset, Dict]:
    """
    Load and split depression detection data.
    
    Target users are INDEPENDENT: mỗi target_user có chuỗi conversations riêng,
    trong đó họ tương tác với các user khác trên social (không phải target_user
    tương tác với nhau). Split train/val/test là random (stratified theo label).
    
    Args:
        interactions_path: Path to CSV file
            Columns: userID, parentID, timestamp, post_id, conversation_id
        embeddings_path: Path to JSON file
            Format: {"target_user_id": {"post_id": [embedding_vector]}}
        labels_path: Path to JSON file
            Format: {"target_user_id": 0 or 1}
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        split_method: 'stratified' (giữ tỉ lệ label) hoặc 'random'
        seed: Random seed cho split
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    print("Loading data...")
    
    # Load CSV interactions
    interactions_df = pd.read_csv(interactions_path)
    print(f"  Loaded {len(interactions_df)} interactions from CSV")
    
    # Load JSON embeddings
    with open(embeddings_path, 'r') as f:
        embeddings_data = json.load(f)
    print(f"  Loaded embeddings for {len(embeddings_data)} target users")
    
    # Load JSON labels
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    print(f"  Loaded labels for {len(labels_data)} target users")
    
    # Get all unique users from interactions
    all_users = set(interactions_df['userID'].unique())
    # parentID có thể là None/NaN cho root posts
    parent_users = interactions_df['parentID'].dropna().unique()
    all_users.update(parent_users)
    
    # Convert to strings for consistency
    all_users = {str(u) for u in all_users}
    
    # Add target users from labels (in case they don't appear in interactions)
    all_users.update(labels_data.keys())
    
    # Create user mappings
    user_to_idx = {user: idx for idx, user in enumerate(sorted(all_users))}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    n_total_users = len(user_to_idx)
    
    print(f"  Found {n_total_users} unique users")
    
    # Build post_id to embedding mapping and create embedding matrix
    # First, collect all unique post_ids
    all_post_ids = set()
    for target_user, posts in embeddings_data.items():
        all_post_ids.update(posts.keys())
    
    # Also add post_ids from interactions
    all_post_ids.update(interactions_df['post_id'].astype(str).unique())
    
    # Create post_id mapping
    post_id_to_idx = {str(pid): idx for idx, pid in enumerate(sorted(all_post_ids))}
    idx_to_post_id = {idx: pid for pid, idx in post_id_to_idx.items()}
    n_posts = len(post_id_to_idx)
    
    # Determine embedding dimension
    embedding_dim = None
    for target_user, posts in embeddings_data.items():
        for post_id, emb in posts.items():
            embedding_dim = len(emb)
            break
        if embedding_dim:
            break
    
    if embedding_dim is None:
        embedding_dim = 768  # Default BERT dimension
        print(f"  Warning: No embeddings found, using default dim {embedding_dim}")
    
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Total posts: {n_posts}")
    
    # Create embedding matrix
    post_embeddings = np.zeros((n_posts, embedding_dim), dtype=np.float32)
    
    # Fill in embeddings from JSON
    for target_user, posts in embeddings_data.items():
        for post_id, emb in posts.items():
            post_id_str = str(post_id)
            if post_id_str in post_id_to_idx:
                post_embeddings[post_id_to_idx[post_id_str]] = np.array(emb)
    
    # Count how many posts have embeddings
    n_with_embeddings = np.sum(np.any(post_embeddings != 0, axis=1))
    print(f"  Posts with embeddings: {n_with_embeddings}/{n_posts}")
    
    # Group interactions by conversation
    conversations_dict = defaultdict(list)
    
    for _, row in interactions_df.iterrows():
        conv_id = str(row['conversation_id'])
        user_id = str(row['userID'])
        parent_id = row['parentID']
        
        # Skip if parent is NaN (root post - no reply relationship)
        if pd.isna(parent_id):
            continue
        
        parent_id = str(parent_id)
        post_id = str(row['post_id'])
        timestamp = float(row['timestamp'])
        
        # Only add if both users exist in mapping
        if user_id in user_to_idx and parent_id in user_to_idx:
            conversations_dict[conv_id].append({
                'source': user_to_idx[user_id],      # User who replied
                'dest': user_to_idx[parent_id],      # User being replied to
                'timestamp': timestamp,
                'post_id': post_id_to_idx.get(post_id, 0)  # Map to embedding index
            })
    
    # Create Conversation objects
    all_conversations = {}
    for conv_id, interactions in conversations_dict.items():
        if len(interactions) == 0:
            continue
        
        # Sort by timestamp
        interactions = sorted(interactions, key=lambda x: x['timestamp'])
        
        conv = Conversation(
            conversation_id=conv_id,
            source_users=np.array([i['source'] for i in interactions]),
            dest_users=np.array([i['dest'] for i in interactions]),
            timestamps=np.array([i['timestamp'] for i in interactions], dtype=np.float64),
            post_ids=np.array([i['post_id'] for i in interactions], dtype=np.int64)
        )
        all_conversations[conv_id] = conv
    
    print(f"  Created {len(all_conversations)} conversations")
    
    # Group conversations by TARGET USER
    user_conversations = defaultdict(list)
    
    for conv_id, conv in all_conversations.items():
        # Add conversation to all target users who participate in it
        for user_idx in conv.unique_users:
            user_str = idx_to_user[user_idx]
            if user_str in labels_data:
                user_conversations[user_idx].append(conv)
    
    # Create UserData objects for target users
    all_user_data = []
    for target_user_str, label in labels_data.items():
        if target_user_str not in user_to_idx:
            print(f"  Warning: Target user {target_user_str} not found in interactions")
            continue
        
        user_idx = user_to_idx[target_user_str]
        convs = user_conversations.get(user_idx, [])
        # IMPORTANT: Do NOT sort conversations here.
        # The conversation sequence is assumed to follow the order in the input data.
        
        user_data = UserData(
            user_id=user_idx,
            user_id_str=target_user_str,
            conversations=convs,
            label=int(label)
        )
        all_user_data.append(user_data)
    
    print(f"  Created {len(all_user_data)} target user samples")
    
    # Print some statistics
    users_with_convs = sum(1 for u in all_user_data if u.n_conversations > 0)
    total_convs = sum(u.n_conversations for u in all_user_data)
    total_interactions = sum(u.total_interactions for u in all_user_data)
    
    print(f"  Users with conversations: {users_with_convs}/{len(all_user_data)}")
    print(f"  Total conversations (assigned): {total_convs}")
    print(f"  Total interactions: {total_interactions}")
    
    # Split data: RANDOM (stratified by label), không chronological
    # Target users độc lập, không cần split theo thời gian
    n_total = len(all_user_data)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    
    rng = np.random.RandomState(seed)
    
    if split_method == "stratified":
        # Stratified split: giữ tỉ lệ depression/non-depression trong mỗi split
        labels_arr = np.array([u.label for u in all_user_data])
        indices = np.arange(n_total)
        
        train_idx, temp_idx = _stratified_split(indices, labels_arr, n_train / n_total, rng)
        val_idx, test_idx = _stratified_split(temp_idx, labels_arr[temp_idx], n_val / len(temp_idx), rng)
        
        train_users = [all_user_data[i] for i in train_idx]
        val_users = [all_user_data[i] for i in val_idx]
        test_users = [all_user_data[i] for i in test_idx]
    else:
        # Random split
        perm = rng.permutation(n_total)
        train_users = [all_user_data[i] for i in perm[:n_train]]
        val_users = [all_user_data[i] for i in perm[n_train:n_train + n_val]]
        test_users = [all_user_data[i] for i in perm[n_train + n_val:]]
    
    print(f"  Split ({split_method}): {len(train_users)} train, {len(val_users)} val, {len(test_users)} test users")
    
    # Create datasets
    train_dataset = DepressionDataset(
        users=train_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    val_dataset = DepressionDataset(
        users=val_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    test_dataset = DepressionDataset(
        users=test_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    metadata = {
        'n_total_users': n_total_users,
        'n_target_users': len(all_user_data),
        'n_posts': n_posts,
        'embedding_dim': embedding_dim,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'post_id_to_idx': post_id_to_idx,
        'idx_to_post_id': idx_to_post_id
    }
    
    return train_dataset, val_dataset, test_dataset, metadata


def load_depression_data_from_parquet_folders(
    data_dir: str,
    neg_folder: str = "neg",
    pos_folder: str = "pos",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_method: str = "stratified",
    seed: int = 42
) -> Tuple[DepressionDataset, DepressionDataset, DepressionDataset, Dict]:
    """
    Load data từ 2 folder neg (label 0) và pos (label 1).
    
    Mỗi folder chứa các file .parquet; tên file (stem) = target user id.
    Mỗi parquet: userID, parentID, timestamp, post_id, conversation_id, embedding.
    
    Args:
        data_dir: Thư mục gốc chứa neg/ và pos/
        neg_folder: Tên folder cho label 0 (mặc định "neg")
        pos_folder: Tên folder cho label 1 (mặc định "pos")
        val_ratio, test_ratio: Tỉ lệ val/test
        split_method: 'stratified' hoặc 'random'
        seed: Random seed
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    data_dir = Path(data_dir)
    neg_path = data_dir / neg_folder
    pos_path = data_dir / pos_folder
    
    if not neg_path.exists():
        raise FileNotFoundError(f"Folder not found: {neg_path}")
    if not pos_path.exists():
        raise FileNotFoundError(f"Folder not found: {pos_path}")
    
    print("Loading data from parquet folders...")
    
    # Thu thập tất cả parquet paths và labels
    parquet_files: List[Tuple[Path, int]] = []
    for p in neg_path.glob("*.parquet"):
        parquet_files.append((p, 0))
    for p in pos_path.glob("*.parquet"):
        parquet_files.append((p, 1))
    
    if len(parquet_files) == 0:
        raise ValueError(f"No .parquet files found in {neg_path} or {pos_path}")
    
    print(f"  Found {len(parquet_files)} parquet files")
    
    # Pass 1: Thu thập tất cả users và (post_id, embedding)
    all_users: set = set()
    post_id_to_embedding: Dict[str, np.ndarray] = {}
    
    for parquet_path, _ in parquet_files:
        df = pd.read_parquet(parquet_path)
        # Cột: userID, parentID, timestamp, post_id, conversation_id, embedding
        for col in ["userID", "parentID", "timestamp", "post_id", "conversation_id", "embedding"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {parquet_path}")
        
        for _, row in df.iterrows():
            if pd.isna(row["userID"]):
                continue
            uid = str(row["userID"])
            all_users.add(uid)
            pid = row["parentID"]
            if pd.notna(pid):
                all_users.add(str(pid))
            
            post_id = str(row["post_id"])
            if post_id not in post_id_to_embedding:
                post_id_to_embedding[post_id] = _embedding_to_1d(row["embedding"])
    
    # User mappings
    user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    n_total_users = len(user_to_idx)
    
    # Post embeddings matrix
    post_ids_sorted = sorted(post_id_to_embedding.keys())
    post_id_to_idx = {pid: i for i, pid in enumerate(post_ids_sorted)}
    idx_to_post_id = {i: pid for pid, i in post_id_to_idx.items()}
    n_posts = len(post_ids_sorted)
    embedding_dim = 0
    for pid in post_ids_sorted:
        arr = np.asarray(post_id_to_embedding[pid], dtype=np.float32).reshape(-1)
        embedding_dim = max(embedding_dim, arr.size)
    post_embeddings = np.zeros((n_posts, embedding_dim), dtype=np.float32)
    for i, pid in enumerate(post_ids_sorted):
        arr = np.asarray(post_id_to_embedding[pid], dtype=np.float32).reshape(-1)
        d = arr.size
        if d >= embedding_dim:
            post_embeddings[i] = arr[:embedding_dim]
        else:
            post_embeddings[i, :d] = arr
    print(f"  Total users: {n_total_users}, posts: {n_posts}, embedding_dim: {embedding_dim}")
    
    # Pass 2: Build UserData cho mỗi parquet
    all_user_data: List[UserData] = []
    
    for parquet_path, label in parquet_files:
        target_user_id_str = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        
        # Đảm bảo target user có trong user_to_idx (có thể chưa xuất hiện trong interactions)
        if target_user_id_str not in user_to_idx:
            user_to_idx[target_user_id_str] = n_total_users
            idx_to_user[n_total_users] = target_user_id_str
            n_total_users += 1
        
        user_idx = user_to_idx[target_user_id_str]
        
        # Bỏ dòng không có parentID (root post) hoặc userID
        df = df[df["parentID"].notna() & df["userID"].notna()].copy()
        if len(df) == 0:
            all_user_data.append(UserData(
                user_id=user_idx,
                user_id_str=target_user_id_str,
                conversations=[],
                label=label
            ))
            continue
        
        # Group by conversation_id
        conversations_list: List[Conversation] = []
        # IMPORTANT: Keep conversation order as it appears in the file (no sorting by conversation_id).
        for conv_id, group in df.groupby("conversation_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            rows_valid = []
            for _, row in group.iterrows():
                uid_str = str(row["userID"])
                pid_str = str(row["parentID"])
                if uid_str not in user_to_idx or pid_str not in user_to_idx:
                    continue
                post_id_str = str(row["post_id"])
                post_idx = post_id_to_idx.get(post_id_str, 0)
                rows_valid.append((
                    user_to_idx[uid_str],
                    user_to_idx[pid_str],
                    float(row["timestamp"]),
                    post_idx
                ))
            if len(rows_valid) == 0:
                continue
            
            source_users = np.array([r[0] for r in rows_valid], dtype=np.int64)
            dest_users = np.array([r[1] for r in rows_valid], dtype=np.int64)
            timestamps = np.array([r[2] for r in rows_valid], dtype=np.float64)
            post_ids = np.array([r[3] for r in rows_valid], dtype=np.int64)
            
            conv = Conversation(
                conversation_id=str(conv_id),
                source_users=source_users,
                dest_users=dest_users,
                timestamps=timestamps,
                post_ids=post_ids
            )
            conversations_list.append(conv)
        # IMPORTANT: Do NOT sort conversations_list.
        # The conversation sequence is assumed to follow the order in the input data.
        
        user_data = UserData(
            user_id=user_idx,
            user_id_str=target_user_id_str,
            conversations=conversations_list,
            label=label
        )
        all_user_data.append(user_data)
    
    print(f"  Created {len(all_user_data)} target user samples")
    
    # Split: test_ratio=0 → chỉ train/val; val_ratio=0 & test_ratio=0 → toàn bộ là train (để dùng làm test set)
    n_total = len(all_user_data)
    if test_ratio <= 0 and val_ratio <= 0:
        n_test = 0
        n_val = 0
        n_train = n_total
    elif test_ratio <= 0:
        n_test = 0
        n_val = max(1, int(n_total * val_ratio)) if n_total >= 2 else 0
        n_train = n_total - n_val
    else:
        n_test = max(1, int(n_total * test_ratio)) if n_total >= 3 else 0
        n_val = max(1, int(n_total * val_ratio)) if n_total >= 2 else 0
        n_train = n_total - n_val - n_test
        if n_train < 0:
            n_train = n_total - 2
            n_val = 1
            n_test = 1
    
    rng = np.random.RandomState(seed)
    labels_arr = np.array([u.label for u in all_user_data])
    indices = np.arange(n_total)
    
    if split_method == "stratified":
        train_idx, temp_idx = _stratified_split(indices, labels_arr, n_train / n_total, rng)
        temp_idx = np.asarray(temp_idx, dtype=np.intp)
        if n_test > 0 and len(temp_idx) > 0:
            val_ratio_rest = n_val / len(temp_idx)
            val_idx, test_idx = _stratified_split(temp_idx, labels_arr[temp_idx], val_ratio_rest, rng)
            test_users = [all_user_data[i] for i in test_idx]
        else:
            val_idx = temp_idx
            test_idx = np.array([], dtype=np.intp)
            test_users = []
        train_users = [all_user_data[i] for i in train_idx]
        val_users = [all_user_data[i] for i in val_idx]
    else:
        perm = rng.permutation(n_total)
        train_users = [all_user_data[i] for i in perm[:n_train]]
        val_users = [all_user_data[i] for i in perm[n_train:n_train + n_val]]
        test_users = [all_user_data[i] for i in perm[n_train + n_val:n_train + n_val + n_test]]
    
    print(f"  Split ({split_method}): {len(train_users)} train, {len(val_users)} val" + (f", {len(test_users)} test" if test_users else " (no test)"))
    
    train_dataset = DepressionDataset(
        users=train_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    val_dataset = DepressionDataset(
        users=val_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    test_dataset = DepressionDataset(
        users=test_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    metadata = {
        'n_total_users': n_total_users,
        'n_target_users': len(all_user_data),
        'n_posts': n_posts,
        'embedding_dim': embedding_dim,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'post_id_to_idx': post_id_to_idx,
        'idx_to_post_id': idx_to_post_id
    }
    
    return train_dataset, val_dataset, test_dataset, metadata


def create_dummy_data(
    n_total_users: int = 100,
    n_target_users: int = 50,
    n_conversations: int = 200,
    avg_interactions: int = 10,
    avg_convs_per_user: int = 5,
    embedding_dim: int = 768,
    depression_ratio: float = 0.3,
    save_dir: Optional[str] = None
) -> Tuple[DepressionDataset, DepressionDataset, DepressionDataset, Dict]:
    """
    Create dummy data for testing.
    
    Args:
        n_total_users: Total number of users in system
        n_target_users: Number of target users (with labels)
        n_conversations: Total number of conversations
        avg_interactions: Average interactions per conversation
        avg_convs_per_user: Average conversations per target user
        embedding_dim: Dimension of post embeddings
        depression_ratio: Ratio of depression labels
        save_dir: Optional directory to save generated data
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    print("Creating dummy data...")
    
    np.random.seed(42)
    
    # Create user mappings
    user_to_idx = {f"user_{i}": i for i in range(n_total_users)}
    idx_to_user = {i: f"user_{i}" for i in range(n_total_users)}
    
    # Select target users
    target_user_indices = np.random.choice(n_total_users, n_target_users, replace=False)
    
    # Assign labels to target users
    target_labels = {}
    for idx in target_user_indices:
        target_labels[idx] = 1 if np.random.random() < depression_ratio else 0
    
    # Generate conversations
    all_conversations = {}
    all_post_embeddings = []
    post_id_counter = 0
    current_time = 0.0
    
    # Track which conversations each target user is in
    user_to_conversations = defaultdict(list)
    
    for conv_idx in range(n_conversations):
        # Random number of interactions
        n_interactions = max(2, int(np.random.exponential(avg_interactions)))
        
        # Random users in this conversation
        n_users_in_conv = min(n_total_users, max(2, np.random.randint(2, 8)))
        
        # Ensure at least one target user is in the conversation
        must_include = np.random.choice(target_user_indices)
        other_users = np.random.choice(
            [u for u in range(n_total_users) if u != must_include],
            n_users_in_conv - 1,
            replace=False
        )
        conv_users = np.concatenate([[must_include], other_users])
        
        # Generate interactions
        source_users = []
        dest_users = []
        timestamps = []
        post_ids = []
        
        for i in range(n_interactions):
            src = np.random.choice(conv_users)
            possible_dests = [u for u in conv_users if u != src]
            if len(possible_dests) == 0:
                possible_dests = conv_users.tolist()
            dst = np.random.choice(possible_dests)
            
            source_users.append(src)
            dest_users.append(dst)
            timestamps.append(current_time + i * np.random.uniform(60, 3600))
            post_ids.append(post_id_counter)
            
            # Generate random embedding
            all_post_embeddings.append(np.random.randn(embedding_dim))
            post_id_counter += 1
        
        current_time = timestamps[-1] + np.random.uniform(3600, 86400)
        
        conv = Conversation(
            conversation_id=f"conv_{conv_idx}",
            source_users=np.array(source_users),
            dest_users=np.array(dest_users),
            timestamps=np.array(timestamps),
            post_ids=np.array(post_ids)
        )
        all_conversations[conv.conversation_id] = conv
        
        # Track which target users are in this conversation
        for user_idx in conv.unique_users:
            if user_idx in target_labels:
                user_to_conversations[user_idx].append(conv)
    
    post_embeddings = np.array(all_post_embeddings, dtype=np.float32)
    print(f"  Generated {len(all_conversations)} conversations with {len(post_embeddings)} posts")
    
    # Create UserData objects
    all_user_data = []
    for user_idx, label in target_labels.items():
        convs = user_to_conversations.get(user_idx, [])
        # IMPORTANT: Do NOT sort conversations here.
        # The conversation sequence is assumed to follow the generation / input order.
        
        user_data = UserData(
            user_id=user_idx,
            user_id_str=idx_to_user[user_idx],
            conversations=convs,
            label=label
        )
        all_user_data.append(user_data)
    
    # Conversations của mỗi user đã sort theo time (trong UserData)
    print(f"  Created {len(all_user_data)} target user samples")
    
    # Save if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV (userID, parentID, timestamp, post_id, conversation_id)
        interactions_data = []
        for conv in all_conversations.values():
            for i in range(conv.n_interactions):
                interactions_data.append({
                    'userID': idx_to_user[conv.source_users[i]],
                    'parentID': idx_to_user[conv.dest_users[i]],
                    'timestamp': conv.timestamps[i],
                    'post_id': f"post_{conv.post_ids[i]}",
                    'conversation_id': conv.conversation_id
                })
        
        interactions_df = pd.DataFrame(interactions_data)
        interactions_df.to_csv(save_dir / 'interactions.csv', index=False)
        
        # Save embeddings JSON: {"target_user_id": {"post_id": [...]}}
        embeddings_json = {}
        for user_data in all_user_data:
            user_str = user_data.user_id_str
            embeddings_json[user_str] = {}
            
            for conv in user_data.conversations:
                for i in range(conv.n_interactions):
                    post_idx = conv.post_ids[i]
                    post_id_str = f"post_{post_idx}"
                    embeddings_json[user_str][post_id_str] = post_embeddings[post_idx].tolist()
        
        with open(save_dir / 'embeddings.json', 'w') as f:
            json.dump(embeddings_json, f)
        
        # Save labels JSON: {"target_user_id": 0 or 1}
        labels_json = {}
        for user_data in all_user_data:
            labels_json[user_data.user_id_str] = user_data.label
        
        with open(save_dir / 'labels.json', 'w') as f:
            json.dump(labels_json, f)
        
        print(f"  Saved data to {save_dir}")
    
    # Split data (70-15-15): stratified random, target users độc lập
    n_total = len(all_user_data)
    n_test = int(n_total * 0.15)
    n_val = int(n_total * 0.15)
    n_train = n_total - n_val - n_test
    
    rng = np.random.RandomState(42)
    labels_arr = np.array([u.label for u in all_user_data])
    indices = np.arange(n_total)
    
    train_idx, temp_idx = _stratified_split(indices, labels_arr, n_train / n_total, rng)
    val_ratio_rest = n_val / len(temp_idx) if len(temp_idx) > 0 else 0.5
    val_idx, test_idx = _stratified_split(temp_idx, labels_arr[temp_idx], val_ratio_rest, rng)
    
    train_users = [all_user_data[i] for i in train_idx]
    val_users = [all_user_data[i] for i in val_idx]
    test_users = [all_user_data[i] for i in test_idx]
    
    # Create datasets
    train_dataset = DepressionDataset(
        users=train_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    val_dataset = DepressionDataset(
        users=val_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    test_dataset = DepressionDataset(
        users=test_users,
        post_embeddings=post_embeddings,
        n_total_users=n_total_users,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )
    
    metadata = {
        'n_total_users': n_total_users,
        'n_target_users': n_target_users,
        'n_posts': len(post_embeddings),
        'embedding_dim': embedding_dim,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user
    }
    
    return train_dataset, val_dataset, test_dataset, metadata

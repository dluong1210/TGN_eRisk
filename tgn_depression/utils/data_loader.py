"""
Data loading utilities for TGN Depression Detection.

Supports: Parquet folders (data_dir/neg/, data_dir/pos/, one .parquet per target user);
create_dummy_data() for testing.
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


def _ego_nodes_from_conv_rows(
    rows: List[Tuple[str, str, float]],
    target_user_str: str,
    max_hops: int,
) -> set:
    """
    Tập user (str) trong L-hop ego của target_user từ danh sách cạnh (uid, pid, ts).
    Dùng khi load parquet để chỉ giữ event trong ego, tránh load event không dùng (model chỉ dùng 0/1/2 layer).
    """
    if max_hops < 0:
        return set()
    ego: set = {target_user_str}
    if max_hops == 0:
        return ego
    n_hops = min(max_hops, 2)
    for _ in range(n_hops):
        next_ego = set(ego)
        for r in rows:
            uid, pid = r[0], r[1]
            if uid in ego:
                next_ego.add(pid)
            if pid in ego:
                next_ego.add(uid)
        ego = next_ego
    return ego


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


def load_depression_data_from_parquet_folders(
    data_dir: str,
    neg_folder: str = "neg",
    pos_folder: str = "pos",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_method: str = "stratified",
    seed: int = 42,
    max_ego_hops: Optional[int] = 2,
    verbose: bool = True,
    drop_none_embeddings: bool = False,
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
        max_ego_hops: Nếu set (0, 1, hoặc 2), chỉ giữ event trong L-hop ego của target user
            trong mỗi conversation, bỏ event không dùng tới (model chỉ dùng 0/1/2 layer).
            None = giữ toàn bộ event (backward compatible).
    
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
    
    if verbose:
        print("Loading data from parquet folders...")
    
    # Thu thập tất cả parquet paths và labels
    parquet_files: List[Tuple[Path, int]] = []
    for p in neg_path.glob("*.parquet"):
        parquet_files.append((p, 0))
    for p in pos_path.glob("*.parquet"):
        parquet_files.append((p, 1))
    
    if len(parquet_files) == 0:
        raise ValueError(f"No .parquet files found in {neg_path} or {pos_path}")
    
    if verbose:
        print(f"  Found {len(parquet_files)} parquet files")
    if verbose and max_ego_hops is not None:
        print(f"  max_ego_hops={max_ego_hops}: chỉ load event trong {max_ego_hops}-hop ego của target user (bỏ event thừa)")
    
    # Một lần đọc mỗi parquet: vừa thu thập users/embeddings vừa lưu raw rows để sau build UserData
    all_users: set = set()
    post_id_to_embedding: Dict[str, np.ndarray] = {}
    # (target_user_id_str, label, list of (uid_str, pid_str, timestamp, post_id_str, conv_id))
    pending_per_file: List[Tuple[str, int, List[Tuple[str, str, float, str, str]]]] = []
    
    for parquet_path, label in parquet_files:
        df = pd.read_parquet(parquet_path)
        target_user_id_str = parquet_path.stem
        all_users.add(target_user_id_str)
        
        for col in ["userID", "parentID", "timestamp", "post_id", "conversation_id", "embedding"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {parquet_path}")
        
        df = df[df["parentID"].notna() & df["userID"].notna()].copy()
        file_rows: List[Tuple[str, str, float, str, str]] = []
        
        # Group by conversation; nếu max_ego_hops set thì chỉ giữ event trong L-hop ego của target user
        conv_groups = list(df.groupby("conversation_id", sort=False))
        for conv_id, group in conv_groups:
            group = group.sort_values("timestamp").reset_index(drop=True)
            # (uid, pid, ts, post_id, embedding) để sau khi lọc ego vẫn có embedding
            rows_raw = []
            for _, row in group.iterrows():
                emb = _embedding_to_1d(row["embedding"])
                # Khi test có thể có embedding=None → emb rỗng.
                # Nếu drop_none_embeddings=True thì bỏ hoàn toàn những event này.
                if drop_none_embeddings and emb.size == 0:
                    continue
                rows_raw.append(
                    (
                        str(row["userID"]),
                        str(row["parentID"]),
                        float(row["timestamp"]),
                        str(row["post_id"]),
                        emb,
                    )
                )
            if max_ego_hops is not None:
                ego = _ego_nodes_from_conv_rows(
                    [(r[0], r[1], r[2]) for r in rows_raw],
                    target_user_id_str,
                    max_ego_hops,
                )
                rows_raw = [r for r in rows_raw if r[0] in ego or r[1] in ego]
            for uid, pid, ts, post_id, emb in rows_raw:
                all_users.add(uid)
                all_users.add(pid)
                if post_id not in post_id_to_embedding:
                    post_id_to_embedding[post_id] = emb
                file_rows.append((uid, pid, ts, post_id, str(conv_id)))
        
        pending_per_file.append((target_user_id_str, label, file_rows))
    
    # User mappings và post embedding matrix (chỉ từ dữ liệu đã đọc)
    user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    n_total_users = len(user_to_idx)
    
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
    if verbose:
        print(f"  Total users: {n_total_users}, posts: {n_posts}, embedding_dim: {embedding_dim} (single-pass load)")
    
    # Build UserData từ pending rows (không đọc lại parquet)
    all_user_data: List[UserData] = []
    for target_user_id_str, label, file_rows in pending_per_file:
        user_idx = user_to_idx[target_user_id_str]
        if len(file_rows) == 0:
            # Không còn event nào cho user này (vd. tất cả embedding=None và bị drop).
            # VẪN giữ nguyên label gốc từ data; chỉ là không có conversation nào.
            all_user_data.append(
                UserData(
                    user_id=user_idx,
                    user_id_str=target_user_id_str,
                    conversations=[],
                    label=label,
                )
            )
            continue
        
        # Group by conversation_id, giữ thứ tự xuất hiện trong file
        conv_id_to_rows: Dict[str, List[Tuple[str, str, float, str]]] = {}
        for uid_str, pid_str, ts, post_id_str, conv_id in file_rows:
            conv_id_to_rows.setdefault(conv_id, []).append((uid_str, pid_str, ts, post_id_str))
        # Thứ tự conv theo lần gặp đầu trong file_rows
        conv_order = []
        seen = set()
        for _, _, _, _, conv_id in file_rows:
            if conv_id not in seen:
                seen.add(conv_id)
                conv_order.append(conv_id)
        
        conversations_list: List[Conversation] = []
        for conv_id in conv_order:
            rows = conv_id_to_rows[conv_id]
            rows = sorted(rows, key=lambda r: r[2])
            rows_valid = []
            for uid_str, pid_str, ts, post_id_str in rows:
                if uid_str not in user_to_idx or pid_str not in user_to_idx:
                    continue
                post_idx = post_id_to_idx.get(post_id_str, 0)
                rows_valid.append((
                    user_to_idx[uid_str],
                    user_to_idx[pid_str],
                    ts,
                    post_idx
                ))
            if len(rows_valid) == 0:
                continue
            conv = Conversation(
                conversation_id=conv_id,
                source_users=np.array([r[0] for r in rows_valid], dtype=np.int64),
                dest_users=np.array([r[1] for r in rows_valid], dtype=np.int64),
                timestamps=np.array([r[2] for r in rows_valid], dtype=np.float64),
                post_ids=np.array([r[3] for r in rows_valid], dtype=np.int64)
            )
            conversations_list.append(conv)
        
        user_data = UserData(
            user_id=user_idx,
            user_id_str=target_user_id_str,
            conversations=conversations_list,
            label=label,
        )
        all_user_data.append(user_data)
    
    if verbose:
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
    
    if verbose:
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

        user_data = UserData(
            user_id=user_idx,
            user_id_str=idx_to_user[user_idx],
            conversations=convs,
            label=label
        )
        all_user_data.append(user_data)

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

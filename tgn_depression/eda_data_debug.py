"""
EDA & debug script: kiểm tra data loading (ego vs no-ego), userID/parentID, và unit test các hàm.
Không chia train/val/test; chỉ thống kê và test.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Cho phép chạy từ repo root: python tgn_depression/eda_data_debug.py
_script_dir = Path(__file__).resolve().parent
if _script_dir not in sys.path:
    sys.path.insert(0, str(_script_dir))


def eda_raw_parquet(data_dir: str, neg_folder: str = "neg", pos_folder: str = "pos", max_files: int = 50):
    """
    EDA trực tiếp trên parquet: thống kê userID, parentID, post_id; kiểm tra parentID có trùng post_id không.
    """
    data_dir = Path(data_dir)
    neg_path = data_dir / neg_folder
    pos_path = data_dir / pos_folder
    if not neg_path.exists() or not pos_path.exists():
        print(f"Skip raw EDA: {neg_path} or {pos_path} not found")
        return

    parquet_files = list(neg_path.glob("*.parquet")) + list(pos_path.glob("*.parquet"))
    if max_files and len(parquet_files) > max_files:
        parquet_files = parquet_files[:max_files]

    all_user_ids = []
    all_parent_ids = []
    all_post_ids = []
    parent_eq_post_count = 0
    parent_in_post_set_count = 0
    total_rows = 0
    file_count = 0

    for p in parquet_files:
        df = pd.read_parquet(p)
        df = df[df["parentID"].notna() & df["userID"].notna()]
        if df.empty:
            continue
        file_count += 1
        uids = df["userID"].astype(str).tolist()
        pids = df["parentID"].astype(str).tolist()
        post_ids = df["post_id"].astype(str).tolist()
        all_user_ids.extend(uids)
        all_parent_ids.extend(pids)
        all_post_ids.extend(post_ids)
        total_rows += len(df)
        for i in range(len(df)):
            pid_val = pids[i]
            post_val = post_ids[i]
            if pid_val == post_val:
                parent_eq_post_count += 1
        post_set = set(post_ids)
        for pid in pids:
            if pid in post_set:
                parent_in_post_set_count += 1

    distinct_user_id = len(set(all_user_ids))
    distinct_parent_id = len(set(all_parent_ids))
    distinct_post_id = len(set(all_post_ids))
    overlap_uid_pid = len(set(all_user_ids) & set(all_parent_ids))
    overlap_pid_post = len(set(all_parent_ids) & set(all_post_ids))

    stats = {
        "n_parquet_files_sampled": file_count,
        "total_rows": total_rows,
        "distinct_userID": distinct_user_id,
        "distinct_parentID": distinct_parent_id,
        "distinct_post_id": distinct_post_id,
        "overlap_userID_and_parentID": overlap_uid_pid,
        "overlap_parentID_and_post_id": overlap_pid_post,
        "parentID_equals_post_id_same_row_count": parent_eq_post_count,
        "parentID_in_post_set_count": parent_in_post_set_count,
    }
    print("EDA Raw Parquet (sample):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def compare_load_with_and_without_ego(
    data_dir: str,
    neg_folder: str = "neg",
    pos_folder: str = "pos",
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
):
    """
    Load 2 lần: max_ego_hops=None và max_ego_hops=2 (không chia train/val/test).
    So sánh n_total_users, n_target_users, tổng số interactions.
    """
    from utils.data_loader import load_depression_data_from_parquet_folders

    def total_interactions(dataset):
        return sum(u.total_interactions for u in dataset.users)

    # Load KHÔNG ego (giữ toàn bộ event)
    train_full, _, _, meta_full = load_depression_data_from_parquet_folders(
        data_dir=data_dir,
        neg_folder=neg_folder,
        pos_folder=pos_folder,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_ego_hops=None,
        verbose=False,
    )
    n_total_no_ego = meta_full["n_total_users"]
    n_target_no_ego = meta_full["n_target_users"]
    n_inter_no_ego = total_interactions(train_full)

    # Load CÓ ego 1-hop
    train_ego1, _, _, meta_ego1 = load_depression_data_from_parquet_folders(
        data_dir=data_dir,
        neg_folder=neg_folder,
        pos_folder=pos_folder,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_ego_hops=1,
        verbose=False,
    )
    n_total_ego1 = meta_ego1["n_total_users"]
    n_inter_ego1 = total_interactions(train_ego1)

    # Load CÓ ego 2-hop
    train_ego2, _, _, meta_ego2 = load_depression_data_from_parquet_folders(
        data_dir=data_dir,
        neg_folder=neg_folder,
        pos_folder=pos_folder,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_ego_hops=2,
        verbose=False,
    )
    n_total_ego2 = meta_ego2["n_total_users"]
    n_inter_ego2 = total_interactions(train_ego2)

    comparison = {
        "no_ego": {"n_total_users": n_total_no_ego, "n_target_users": n_target_no_ego, "total_interactions": n_inter_no_ego},
        "ego_1hop": {"n_total_users": n_total_ego1, "total_interactions": n_inter_ego1},
        "ego_2hop": {"n_total_users": n_total_ego2, "n_target_users": n_target_no_ego, "total_interactions": n_inter_ego2},
    }
    print("So sánh load: no ego vs ego 1-hop vs ego 2-hop (không chia train/val/test):")
    print(f"  No ego:    n_total_users={n_total_no_ego}, n_target_users={n_target_no_ego}, total_interactions={n_inter_no_ego}")
    print(f"  Ego 1hop:  n_total_users={n_total_ego1}, total_interactions={n_inter_ego1}  (chênh vs no_ego: users {n_total_no_ego - n_total_ego1}, interactions {n_inter_no_ego - n_inter_ego1})")
    print(f"  Ego 2hop:  n_total_users={n_total_ego2}, total_interactions={n_inter_ego2}  (chênh vs no_ego: users {n_total_no_ego - n_total_ego2}, interactions {n_inter_no_ego - n_inter_ego2})")

    return comparison


def test_ego_nodes_from_conv_rows():
    """Unit test _ego_nodes_from_conv_rows."""
    from utils.data_loader import _ego_nodes_from_conv_rows

    # 0-hop: chỉ target
    rows = [("A", "B", 1.0), ("B", "C", 2.0)]
    ego0 = _ego_nodes_from_conv_rows(rows, "A", 0)
    assert ego0 == {"A"}, f"0-hop expected {{A}}, got {ego0}"

    # 1-hop: A và B (A-B edge)
    ego1 = _ego_nodes_from_conv_rows(rows, "A", 1)
    assert ego1 == {"A", "B"}, f"1-hop expected {{A,B}}, got {ego1}"

    # 2-hop: A, B, C
    ego2 = _ego_nodes_from_conv_rows(rows, "A", 2)
    assert ego2 == {"A", "B", "C"}, f"2-hop expected {{A,B,C}}, got {ego2}"

    # Target không xuất hiện trong rows vẫn có trong ego
    ego1b = _ego_nodes_from_conv_rows([("B", "C", 1.0)], "A", 1)
    assert ego1b == {"A"}, f"target not in rows: expected {{A}}, got {ego1b}"

    print("  test_ego_nodes_from_conv_rows: PASSED")
    return True


def test_stratified_split():
    """Unit test _stratified_split: tỉ lệ class được giữ trong train."""
    from utils.data_loader import _stratified_split

    rng = np.random.RandomState(42)
    indices = np.arange(100)
    labels = np.array([0] * 70 + [1] * 30)  # 70% class 0, 30% class 1
    train_idx, rest_idx = _stratified_split(indices, labels, train_ratio=0.7, rng=rng)
    train_labels = labels[train_idx]
    n0 = (train_labels == 0).sum()
    n1 = (train_labels == 1).sum()
    # Train nên ~70% of 70 = 49 class 0, ~70% of 30 = 21 class 1
    assert abs(n0 - 49) <= 2 and abs(n1 - 21) <= 2, f"stratified train: got {n0} class0, {n1} class1"
    print("  test_stratified_split: PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="EDA & debug data loading (no train/val/test split).")
    parser.add_argument("--data_dir", type=str, default="C:\\Users\\7420\\Code\\Python\\TGN_eRisk\\prune\\output\\versions\\3\\eRisk2025_mapped_pruned", help="Data directory (neg/pos folders).")
    parser.add_argument("--neg_folder", type=str, default="neg")
    parser.add_argument("--pos_folder", type=str, default="pos")
    parser.add_argument("--max_parquet_files", type=int, default=50, help="Max parquet files for raw EDA (0 = all).")
    parser.add_argument("--skip_load", action="store_true", help="Skip full load comparison (only raw EDA + tests).")
    args = parser.parse_args()

    print("=== 1. Unit tests ===")
    try:
        test_ego_nodes_from_conv_rows()
        test_stratified_split()
    except Exception as e:
        print(f"  FAILED: {e}")
        raise

    print("=== 2. EDA raw parquet (userID / parentID / post_id) ===")
    max_files = args.max_parquet_files if args.max_parquet_files > 0 else None  # 0 = all
    try:
        eda_raw_parquet(args.data_dir, args.neg_folder, args.pos_folder, max_files=max_files)
    except Exception as e:
        print(f"  Raw EDA error (data_dir có thể không có parquet): {e}")

    if not args.skip_load:
        print("=== 3. So sánh load: không ego vs ego 2-hop ===")
        try:
            compare_load_with_and_without_ego(
                args.data_dir,
                args.neg_folder,
                args.pos_folder,
                val_ratio=0.0,
                test_ratio=0.0,
            )
        except Exception as e:
            print(f"  Load comparison error: {e}")
            raise
    else:
        print("=== 3. Skip load comparison (--skip_load) ===")

    print("Done.")


if __name__ == "__main__":
    main()

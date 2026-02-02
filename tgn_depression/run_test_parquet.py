"""
Chạy evaluation (test) với data parquet và model đã lưu.

Data: cùng format khi train — thư mục chứa neg/ và pos/, mỗi folder các file .parquet.
Toàn bộ data được dùng làm tập đánh giá (không chia train/val).

Cách chạy:
  python run_test_parquet.py <data_dir> <model_path>
  python run_test_parquet.py /path/to/erisk_mapped ./saved_models/best_model_carryover.pth
  python run_test_parquet.py /path/to/erisk_mapped ./saved_models/best_model_carryover.pth --neg_folder neg --pos_folder pos

Lưu ý: Nên dùng cùng data_dir và format như khi train để tránh lỗi kích thước model (n_users, post_embeddings).
Nếu dùng data khác, model sẽ build theo data đó; load state_dict có thể bỏ qua một số key không khớp.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Đảm bảo import từ tgn_depression
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from model.tgn_sequential import TGNSequential
from utils.data_loader import load_depression_data_from_parquet_folders
from utils.utils import set_seed, get_device
from train_sequential import evaluate, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy test/evaluation với data parquet và saved model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Đường dẫn thư mục chứa neg/ và pos/ (cùng format khi train)",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Đường dẫn file model đã lưu (vd: ./saved_models/best_model_carryover.pth)",
    )
    parser.add_argument("--neg_folder", type=str, default="neg", help="Tên folder label 0")
    parser.add_argument("--pos_folder", type=str, default="pos", help="Tên folder label 1")
    parser.add_argument("--results_path", type=str, default=None,
                        help="Đường dẫn file results JSON (mặc định: cùng thư mục với model, results_<mode>.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("logs")
    logger.info("Run test/evaluation with parquet data and saved model")
    logger.info(f"  data_dir   = {args.data_dir}")
    logger.info(f"  model_path = {args.model_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Không tìm thấy file model: {model_path}")
        sys.exit(1)

    # Suy ra results_path nếu không cho
    if args.results_path is None:
        # best_model_carryover.pth -> results_carryover.json
        stem = model_path.stem  # best_model_carryover
        if stem.startswith("best_model_"):
            mode = stem.replace("best_model_", "")
            results_path = model_path.parent / f"results_{mode}.json"
        else:
            results_path = model_path.parent / "results_carryover.json"
    else:
        results_path = Path(args.results_path)
    if not results_path.exists():
        logger.error(f"Không tìm thấy file results (cần để build model): {results_path}")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)
    train_args = results.get("args", {})
    if not train_args:
        logger.error("File results không chứa 'args'. Cần file results từ lần train.")
        sys.exit(1)

    set_seed(args.seed)
    device = get_device(args.gpu)
    logger.info(f"Device: {device}")

    # Load data: toàn bộ làm một tập (val_ratio=0, test_ratio=0)
    logger.info("Loading data (full set for evaluation)...")
    train_dataset, _val_dataset, _test_dataset, metadata = load_depression_data_from_parquet_folders(
        data_dir=args.data_dir,
        neg_folder=args.neg_folder,
        pos_folder=args.pos_folder,
        val_ratio=0.0,
        test_ratio=0.0,
        split_method="stratified",
        seed=args.seed,
    )
    # train_dataset ở đây chứa toàn bộ mẫu (dùng làm tập test)
    eval_dataset = train_dataset
    logger.info(f"  Số mẫu đánh giá: {len(eval_dataset.users)}")

    # Build model giống lúc train
    from argparse import Namespace
    nargs = Namespace(**{k: v for k, v in train_args.items()})
    model = TGNSequential(
        n_users=metadata["n_total_users"],
        edge_features=eval_dataset.post_embeddings,
        device=device,
        sequence_mode=nargs.sequence_mode,
        lstm_hidden_dim=getattr(nargs, "lstm_hidden_dim", 128),
        lstm_num_layers=getattr(nargs, "lstm_num_layers", 1),
        lstm_bidirectional=getattr(nargs, "lstm_bidirectional", False),
        n_layers=getattr(nargs, "n_layers", 1),
        n_heads=getattr(nargs, "n_heads", 2),
        dropout=getattr(nargs, "dropout", 0.1),
        use_memory=getattr(nargs, "use_memory", True),
        memory_dimension=getattr(nargs, "memory_dim", 172),
        embedding_module_type=getattr(nargs, "embedding_module", "graph_attention"),
        message_function_type=getattr(nargs, "message_function", "identity"),
        aggregator_type=getattr(nargs, "aggregator", "last"),
        memory_updater_type=getattr(nargs, "memory_updater", "gru"),
        n_neighbors=getattr(nargs, "n_neighbors", 10),
        num_classes=2,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Keys không có trong model (bỏ qua): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Keys thừa trong file (bỏ qua): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    n_neighbors = getattr(nargs, "n_neighbors", 10)
    loss, metrics = evaluate(
        model=model,
        dataset=eval_dataset,
        device=device,
        n_neighbors=n_neighbors,
    )

    logger.info("=" * 50)
    logger.info("Kết quả đánh giá (Evaluation / Test)")
    logger.info(f"  Loss:    {loss:.4f}")
    logger.info(f"  Metrics: {metrics}")
    logger.info("=" * 50)
    print("\nEvaluation metrics:", metrics)


if __name__ == "__main__":
    main()

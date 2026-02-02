"""
Training với data parquet (folder neg/ + pos/).

Cách chạy:
  python run_train_parquet.py <data_dir>
  python run_train_parquet.py /kaggle/input/erisk-post-embedding-mapped/eRisk2022_mapped
  python run_train_parquet.py /kaggle/input/erisk-post-embedding-mapped/eRisk2022_mapped --epochs 30 --sequence_mode lstm

data_dir phải chứa 2 folder:
  - neg/  (label 0): các file .parquet, tên file = target_user
  - pos/  (label 1): các file .parquet, tên file = target_user
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo import train_sequential (cùng thư mục tgn_depression)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_sequential import main, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TGN Sequential với data parquet (neg/ + pos/)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Đường dẫn tới thư mục chứa neg/ và pos/ (vd: /kaggle/input/erisk-post-embedding-mapped/eRisk2022_mapped)",
    )
    parser.add_argument("--neg_folder", type=str, default="neg", help="Tên folder label 0")
    parser.add_argument("--pos_folder", type=str, default="pos", help="Tên folder label 1")
    parser.add_argument("--sequence_mode", type=str, default="carryover", choices=["carryover", "lstm"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true", help="Train trên 2 GPU (DDP)")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="GPU IDs khi dùng --multi_gpu (vd: 0,1)")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_num_layers", type=int, default=1)
    parser.add_argument("--lstm_bidirectional", action="store_true")
    parser.add_argument("--memory_dim", type=int, default=172)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--split_method", type=str, default="stratified", choices=["stratified", "random"])
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--embedding_module", type=str, default="graph_attention")
    parser.add_argument("--message_function", type=str, default="identity")
    parser.add_argument("--aggregator", type=str, default="last")
    parser.add_argument("--memory_updater", type=str, default="gru")
    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_false", dest="use_memory")
    args = parser.parse_args()
    return args


def build_train_args(args):
    """Chuyển args từ run_train_parquet sang format của train_sequential.main()."""
    from argparse import Namespace
    return Namespace(
        sequence_mode=args.sequence_mode,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_bidirectional=args.lstm_bidirectional,
        data_dir=args.data_dir,
        data_format="parquet_folders",
        neg_folder=args.neg_folder,
        pos_folder=args.pos_folder,
        use_dummy_data=False,
        save_dummy=False,
        n_total_users=100,
        n_target_users=50,
        n_conversations=200,
        avg_interactions=10,
        embedding_dim=768,
        split_method=args.split_method,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        use_memory=args.use_memory,
        memory_dim=args.memory_dim,
        n_neighbors=args.n_neighbors,
        embedding_module=args.embedding_module,
        message_function=args.message_function,
        aggregator=args.aggregator,
        memory_updater=args.memory_updater,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        gpu=args.gpu,
        multi_gpu=args.multi_gpu,
        gpu_ids=args.gpu_ids,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Lỗi: Thư mục không tồn tại: {data_dir}")
        sys.exit(1)
    neg_path = data_dir / args.neg_folder
    pos_path = data_dir / args.pos_folder
    if not neg_path.exists():
        print(f"Lỗi: Không tìm thấy folder neg: {neg_path}")
        sys.exit(1)
    if not pos_path.exists():
        print(f"Lỗi: Không tìm thấy folder pos: {pos_path}")
        sys.exit(1)

    train_args = build_train_args(args)
    main(train_args)

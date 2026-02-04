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
from utils.metrics import METRICS, Run
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng quyết định (p>=threshold => phát hiện depression) cho metrics eRisk",
    )
    return parser.parse_args()


def build_temporal_run(
    model: TGNSequential,
    dataset,
    device: torch.device,
    n_neighbors: int,
    threshold: float = 0.5,
) -> tuple[Run, dict]:
    """
    Chạy predict theo từng conversation cho từng user và xây dựng cấu trúc Run
    để tính các metrics trong utils.metrics (ERDE, latency, speed, ...).

    Với LSTM mode:
      - Sau mỗi conversation i, thu được embedding của target user.
      - Chuỗi embeddings [1..i] được feed qua LSTM, lấy output tại state i
        (ở đây dùng lstm_out tại mỗi timestep, rồi classification head trên
        toàn bộ sequence một lần) → p_i.
      - Quyết định tại conversation i: decision_i = (p_i >= threshold).

    Run[subject] = [(time_str, conv_id_str, decision_bool, score_float), ...]
    """
    base_model = model
    base_model.eval()

    run: Run = {}
    golden: dict[str, bool] = {}

    with torch.inference_mode():
        for user_data in dataset.users:
            subject_id = user_data.user_id_str
            golden[subject_id] = bool(user_data.label)

            # Nếu không có conversation hoặc không có interaction, không ra quyết định thời gian.
            conversations = user_data.get_conversations_sorted()
            if (
                len(conversations) == 0
                or user_data.total_interactions == 0
            ):
                run[subject_id] = []
                continue

            # Giới hạn số conversation giống forward_lstm để tránh OOM.
            max_convs = getattr(base_model, "max_conversations_per_user", None)
            if max_convs is not None and len(conversations) > max_convs:
                conversations = conversations[-max_convs:]

            # Hiện tại chỉ hỗ trợ LSTM mode cho metrics theo thời gian.
            if getattr(base_model, "sequence_mode", "carryover") != "lstm":
                # Fallback: một quyết định duy nhất tại cuối chuỗi.
                base_model.reset_state()
                logits = base_model.forward_user(user_data, n_neighbors)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                decision = prob >= threshold
                last_conv = conversations[-1]
                run[subject_id] = [
                    (str(last_conv.end_time), str(last_conv.conversation_id), bool(decision), float(prob))
                ]
                continue

            # ========== LSTM mode: tính embedding sau từng conversation ==========
            base_model.reset_state()
            target_user = user_data.user_id
            conversation_embeddings = []
            used_convs = []

            for conv in conversations:
                if conv.n_interactions == 0:
                    continue

                # Reset memory trước mỗi conversation để đảm bảo độc lập,
                # giống như forward_lstm (nhánh tuần tự).
                if base_model.use_memory:
                    base_model.memory.__init_memory__()

                # Process conversation (build neighbor_finder + memory update).
                end_time = base_model.process_conversation(
                    conv, n_neighbors, target_user=target_user
                )

                # Lấy embedding của target user tại thời điểm end_time.
                user_emb = base_model.compute_user_embedding(
                    target_user, end_time, n_neighbors
                )
                conversation_embeddings.append(user_emb)
                used_convs.append(conv)

            if len(conversation_embeddings) == 0:
                run[subject_id] = []
                continue

            # Chuỗi embeddings: [1, n_convs, emb_dim]
            emb_seq = torch.cat(conversation_embeddings, dim=0).unsqueeze(0)

            # Forward LSTM → lstm_out (hidden cho từng timestep).
            lstm_out, _ = base_model.lstm(emb_seq)  # lstm_out: [1, seq_len, hidden_dim*dirs]

            # Classification head trên từng timestep:
            #   - reshape về [seq_len, hidden_dim*dirs]
            #   - classifier xử lý cả batch → logits_seq [seq_len, num_classes]
            logits_seq = base_model.classifier(lstm_out.squeeze(0))
            probs = torch.softmax(logits_seq, dim=1)[:, 1].cpu().numpy()

            decisions = probs >= threshold
            subject_predictions = []

            for i, (conv, dec, p) in enumerate(zip(used_convs, decisions, probs)):
                # time_str dùng end_time; conv_id_str là conversation_id gốc.
                time_str = str(conv.end_time)
                conv_id_str = str(conv.conversation_id)
                subject_predictions.append(
                    (time_str, conv_id_str, bool(dec), float(p))
                )

            run[subject_id] = subject_predictions

    return run, golden


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
    logger.info(f"  Loss (per-user):    {loss:.4f}")
    logger.info(f"  Metrics (per-user): {metrics}")

    # ========== Extra: metrics kiểu eRisk dựa trên quyết định theo từng conversation ==========
    try:
        temporal_run, golden = build_temporal_run(
            model=model,
            dataset=eval_dataset,
            device=device,
            n_neighbors=n_neighbors,
            threshold=args.threshold,
        )

        temporal_metrics: dict[str, float] = {}
        for name, (metric_fn, _lower_is_better) in METRICS.items():
            temporal_metrics[name] = float(metric_fn(temporal_run, golden))

        logger.info("Metrics theo từng conversation (eRisk style):")
        for k, v in temporal_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        print("\nEvaluation metrics (per-user):", metrics)
        print("Temporal metrics (per-conversation / eRisk):", temporal_metrics)
    except Exception as e:  # pragma: no cover - chỉ log, không dừng chương trình
        logger.exception(f"Lỗi khi tính temporal metrics (ERDE/latency): {e}")
        print("\nEvaluation metrics (per-user):", metrics)


if __name__ == "__main__":
    main()

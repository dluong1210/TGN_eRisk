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
    batch_size: int = 1,
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
    
    Args:
        batch_size: Số users xử lý cùng lúc (chỉ áp dụng cho non-LSTM mode).
                   LSTM mode vẫn xử lý từng user để đảm bảo early detection đúng.
    """
    base_model = model
    base_model.eval()

    run: Run = {}
    golden: dict[str, bool] = {}

    with torch.inference_mode():
        # Tối ưu: xử lý users theo batch để giảm overhead
        users_list = list(dataset.users)
        for batch_start in range(0, len(users_list), max(1, batch_size)):
            batch_end = min(batch_start + batch_size, len(users_list))
            batch_users = users_list[batch_start:batch_end]
            
            for user_data in batch_users:
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

                # Khi test: chạy lần lượt toàn bộ conversation, không truncate.

                # ========== Carryover mode: predict ở mọi điểm (embedding từng conversation → classifier) ==========
                if getattr(base_model, "sequence_mode", "carryover") == "carryover":
                    base_model.reset_state()
                    target_user = user_data.user_id
                    # Phase 1: Chạy tuần tự toàn bộ conversation, embedding sau conv i làm init node cho conv i+1; thu list embedding từng bước.
                    conv_embeddings: list = []  # (conv, embedding) cho conv có interactions
                    for conv in conversations:
                        if conv.n_interactions == 0:
                            continue
                        end_time = base_model.process_conversation(
                            conv, n_neighbors, target_user=target_user
                        )
                        user_emb = base_model.compute_user_embedding(
                            target_user, end_time, n_neighbors
                        )
                        conv_embeddings.append((conv, user_emb))
                        # Carry-over: embedding này làm node feature init cho conversation sau
                        base_model._node_features_custom[target_user] = user_emb.squeeze(0).detach()
                        if base_model.use_memory:
                            base_model.memory.__init_memory__()
                    # Phase 2: Gom embeddings, chạy classifier một lần (giống LSTM: sequence → classifier từng bước)
                    if len(conv_embeddings) == 0:
                        subject_predictions = [
                            (str(c.end_time), str(c.conversation_id), False, 0.0)
                            for c in conversations
                        ]
                    else:
                        emb_stack = torch.cat([e for _, e in conv_embeddings], dim=0)  # [n_valid, dim]
                        logits = base_model.classifier(emb_stack)  # [n_valid, num_classes]
                        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                        decisions = probs >= threshold
                        valid_idx = 0
                        subject_predictions = []
                        for conv in conversations:
                            if conv.n_interactions > 0:
                                subject_predictions.append(
                                    (str(conv.end_time), str(conv.conversation_id),
                                     bool(decisions[valid_idx]), float(probs[valid_idx]))
                                )
                                valid_idx += 1
                            else:
                                subject_predictions.append(
                                    (str(conv.end_time), str(conv.conversation_id), False, 0.0)
                                )
                    run[subject_id] = subject_predictions
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                # ========== LSTM mode: predict ở mọi điểm (mỗi conversation một entry) ==========
                # Run phải có kích thước = số lượng conversation của user. Chạy toàn bộ, không truncate.
                base_model.reset_state()
                target_user = user_data.user_id

                # Danh sách conversation hợp lệ (có interactions) để lấy embedding
                valid_conversations = [c for c in conversations if c.n_interactions > 0]

                if len(valid_conversations) == 0:
                    # Tất cả conversation rỗng → mỗi conv một entry (False, 0.0)
                    subject_predictions = [
                        (str(c.end_time), str(c.conversation_id), False, 0.0)
                        for c in conversations
                    ]
                    run[subject_id] = subject_predictions
                    continue

                # Tạm tắt truncate để chạy toàn bộ conversation
                _saved_max_convs = getattr(base_model, "max_conversations_per_user", None)
                try:
                    base_model.max_conversations_per_user = None
                    conversation_embeddings = base_model.process_conversations_batch(
                        conversations=valid_conversations,
                        target_user=target_user,
                        n_neighbors=n_neighbors,
                        use_larger_batches=True,
                    )
                finally:
                    base_model.max_conversations_per_user = _saved_max_convs

                if len(conversation_embeddings) == 0:
                    subject_predictions = [
                        (str(c.end_time), str(c.conversation_id), False, 0.0)
                        for c in conversations
                    ]
                    run[subject_id] = subject_predictions
                    continue

                # Chuỗi embeddings: [1, n_valid_convs, emb_dim]
                emb_seq = torch.cat(conversation_embeddings, dim=0).unsqueeze(0)

                # Forward LSTM → lstm_out (hidden cho từng timestep).
                lstm_out, _ = base_model.lstm(emb_seq)
                logits_seq = base_model.classifier(lstm_out.squeeze(0))
                probs = torch.softmax(logits_seq, dim=1)[:, 1]
                probs_np = probs.detach().cpu().numpy()
                decisions = probs_np >= threshold

                # Một entry cho mỗi conversation: len(run[subject_id]) == len(conversations)
                # Không truncate nên n_emb == n_valid; conv rỗng → (False, 0.0).
                valid_idx = 0
                subject_predictions = []
                for conv in conversations:
                    if conv.n_interactions > 0:
                        subject_predictions.append(
                            (str(conv.end_time), str(conv.conversation_id),
                             bool(decisions[valid_idx]), float(probs_np[valid_idx]))
                        )
                        valid_idx += 1
                    else:
                        subject_predictions.append(
                            (str(conv.end_time), str(conv.conversation_id), False, 0.0)
                        )

                run[subject_id] = subject_predictions

                # Tối ưu: clear GPU cache sau mỗi user (nếu dùng GPU)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

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
        drop_none_embeddings=True,
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

    # Khởi tạo embedding_module trước khi load state_dict để tránh key thừa
    # (embedding_module được khởi tạo lazy, nếu không init trước thì sẽ không có trong model.state_dict())
    # Lưu ý: neighbor_finder ban đầu là None, nhưng để init embedding_module cần một object
    # (không nhất thiết phải hoạt động, chỉ cần structure đúng để tạo layers/weights)
    from utils.neighbor_finder import get_neighbor_finder
    # Tạo dummy neighbor_finder tối thiểu (1 edge) để init embedding_module
    dummy_sources = np.array([0], dtype=np.int64)
    dummy_dests = np.array([1], dtype=np.int64)
    dummy_timestamps = np.array([0.0], dtype=np.float64)
    dummy_post_ids = np.array([0], dtype=np.int64)
    model.neighbor_finder = get_neighbor_finder(
        sources=dummy_sources,
        destinations=dummy_dests,
        edge_idxs=dummy_post_ids,
        timestamps=dummy_timestamps,
        n_nodes=metadata["n_total_users"],
        uniform=False
    )
    model._init_embedding_module()
    # Sau khi init xong, có thể reset về None nếu muốn (nhưng không cần thiết)

    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Keys không có trong model (bỏ qua): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Keys thừa trong file (bỏ qua): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    n_neighbors = getattr(nargs, "n_neighbors", 10)
    loss, metrics, _, _ = evaluate(
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

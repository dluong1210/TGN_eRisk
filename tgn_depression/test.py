"""
Test script for TGNUserSequence (eRisk temporal evaluation).

Per-event prediction: sau mỗi event lấy target_user embedding → classifier.
Output: run format cho METRICS (ERDE, latency, F1, ...).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from model.tgn_user_sequence import TGNUserSequence
from utils.data_structures import DepressionDataset
from utils.data_loader import load_depression_data_from_parquet_folders, create_dummy_data
from utils.utils import get_device
from utils.metrics import METRICS

# Run structure: run[subject_id] = [(time_str, conv_id_str, decision_bool, score_float), ...]
Run = Dict[str, Sequence[Tuple[str, str, bool, float]]]


def build_temporal_run(
    model: TGNUserSequence,
    dataset: DepressionDataset,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Run, Dict[str, bool]]:
    """
    Build temporal run: per-event predictions cho mỗi user.

    Returns:
        run: subject_id -> [(time_str, conv_id_str, decision, score), ...]
        golden: subject_id -> label (bool)
    """
    model.eval()
    run: Run = {}
    golden: Dict[str, bool] = {}

    with torch.inference_mode():
        for user_data in dataset.users:
            subject_id = user_data.user_id_str
            golden[subject_id] = bool(user_data.label)

            if user_data.total_interactions == 0:
                run[subject_id] = [("0", "none", False, 0.5)]
                continue

            model.reset_state()
            predictions = model.forward(user_data, return_per_event=True)

            if len(predictions) == 0:
                run[subject_id] = [("0", "none", False, 0.5)]
                continue

            run_entries = []
            for ts, conv_id, emb in predictions:
                logits = model.classifier(emb.unsqueeze(0))
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                decision = prob >= threshold
                run_entries.append((str(ts), conv_id, decision, float(prob)))

            run[subject_id] = run_entries

    return run, golden


def main():
    parser = argparse.ArgumentParser(description="Test TGNUserSequence (eRisk temporal)")
    parser.add_argument("data_dir", type=str, nargs="?", default="./data", help="Data directory (parent of neg/pos)")
    parser.add_argument("model_path", type=str, nargs="?", default="./saved_models/best_model.pth", help="Path to model .pth")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use in-memory dummy data")
    parser.add_argument("--neg_folder", type=str, default="neg")
    parser.add_argument("--pos_folder", type=str, default="pos")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_ego_hops", type=int, default=2)

    args = parser.parse_args()
    device = get_device(args.gpu)

    print("Loading data...")
    if args.use_dummy_data:
        _, _, test_dataset, metadata = create_dummy_data(
            n_total_users=100,
            n_target_users=20,
            n_conversations=50,
            avg_interactions=10,
            embedding_dim=768,
            depression_ratio=0.3,
            save_dir=None,
        )
    else:
        _, _, test_dataset, metadata = load_depression_data_from_parquet_folders(
            data_dir=args.data_dir,
            neg_folder=args.neg_folder,
            pos_folder=args.pos_folder,
            val_ratio=0.1,
            test_ratio=0.3,
            split_method="stratified",
            max_ego_hops=args.max_ego_hops if args.max_ego_hops >= 0 else None,
            verbose=True,
        )

    if len(test_dataset.users) == 0:
        print("No test users.")
        return

    print("Loading model...")
    model = TGNUserSequence(
        n_users=metadata["n_total_users"],
        edge_features=test_dataset.post_embeddings,
        device=device,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    print("Building temporal run...")
    run, golden = build_temporal_run(
        model=model,
        dataset=test_dataset,
        device=device,
        threshold=args.threshold,
    )

    print("\nMetrics:")
    for name, (metric_fn, _) in METRICS.items():
        val = float(metric_fn(run, golden))
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()

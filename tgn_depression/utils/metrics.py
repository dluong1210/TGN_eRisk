from typing import Callable, Dict, Sequence, Tuple
from functools import partial

import numpy as np

# Run structure:
#   run[subject_id] = [(time_str, conv_id_str, decision_bool, score_float), ...]
Run = Dict[str, Sequence[Tuple[str, str, bool, float]]]
Metric = Callable[[Run], float]


def erde(decisions: Sequence[bool], truth: bool, o: float) -> float:
    c_fn = 1
    c_fp = 98 / 1400  # proportion of positive subjects
    c_tp = 1  # late detection is equivalent to not detecting at all

    def lc_o(k: int) -> float:
        return 1 - 1 / (1 + np.exp(k - o))

    try:
        delay = decisions.index(True) + 1
        decision = True
    except ValueError:
        delay = None
        decision = False

    if decision and not truth:
        return c_fp
    if not decision and truth:
        return c_fn
    if decision and truth:
        return c_tp * lc_o(delay)
    if not decision and not truth:
        return 0.0


def mean_erde(run: Run, golden, *, o: float) -> float:
    results = []
    for subject, predictions in run.items():
        # predictions: (time_str, conv_id_str, decision_bool, score_float)
        decisions = [decision for _, _, decision, _ in predictions]
        results.append(erde(decisions, golden[subject], o))
    return np.mean(results)


mean_erde_5 = partial(mean_erde, o=5)
mean_erde_50 = partial(mean_erde, o=50)


def recall_precision_f1(run: Run, golden) -> Tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    for subject in run:
        decision = any(decision for _, _, decision, _ in run[subject])
        truth = golden[subject]

        if decision and truth:
            tp += 1
        elif decision and not truth:
            fp += 1
        elif not decision and truth:
            fn += 1

    recall = tp / ((tp + fn) or 1)
    precision = tp / ((tp + fp) or 1)
    f1 = 2 * tp / ((2 * tp + fp + fn) or 1)
    return recall, precision, f1


def recall(run: Run, golden) -> float:
    recall, _, _ = recall_precision_f1(run, golden)
    return recall


def precision(run: Run, golden) -> float:
    _, precision, _ = recall_precision_f1(run, golden)
    return precision


def f1(run: Run, golden) -> float:
    _, _, f1 = recall_precision_f1(run, golden)
    return f1


def latency(run: Run, golden) -> float:
    delays = []
    for subject in run:
        decisions = [decision for _, _, decision, _ in run[subject]]
        decision = any(decisions)
        truth = golden[subject]
        if decision and truth:
            delay = decisions.index(True) + 1
            delays.append(delay)
    if len(delays) > 0:
        return np.median(delays)
    return float("inf")


def speed(run: Run, golden) -> float:
    # Magic number to make penalty(median(post_len)) = 0.5
    # p = 0.0017419306441650182
    p = 0.0078

    penalties = []
    for subject in run:
        decisions = [decision for _, _, decision, _ in run[subject]]
        decision = any(decisions)
        truth = golden[subject]
        if decision and truth:
            delay = decisions.index(True) + 1
            penalty = -1 + 2 / (1 + np.exp(-p * (delay - 1)))
            penalties.append(penalty)
    if len(penalties) > 0:
        return 1 - np.median(penalties)
    return 0


def latency_f1(run: Run, golden):
    return f1(run, golden) * speed(run, golden)


METRICS = {
    "erde5": (mean_erde_5, True),
    "erde50": (mean_erde_50, True),
    "precision": (precision, False),
    "recall": (recall, False),
    "f1": (f1, False),
    "latency": (latency, True),
    "speed": (speed, False),
    "latency-f1": (latency_f1, False),
}
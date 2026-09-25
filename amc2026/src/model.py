"""
model.py
LightGBM-based pairwise entity matching model.
Optimized for tabular string similarity features and Macro F_0.5 metric tuning.
"""

import os
import pickle
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import lightgbm as lgb
from evaluate import compute_entity_f05


def train_lgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict] = None
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier on entity-pair feature vectors.
    """
    default_params = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain"
    }
    if params:
        default_params.update(params)

    clf = lgb.LGBMClassifier(**default_params)

    if X_val is not None and y_val is not None:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
    else:
        clf.fit(X_train, y_train)

    return clf


def find_optimal_threshold_for_f05(
    clf: lgb.LGBMClassifier,
    val_s1_ids: List[str],
    s1_to_candidates: Dict[str, List[Tuple[str, np.ndarray]]],
    gt_mapping: Dict[str, Set[str]],
    threshold_range: np.ndarray = np.arange(0.35, 0.90, 0.02)
) -> Tuple[float, float, Dict[float, float]]:
    """
    Evaluates candidate predictions across a grid of thresholds to maximize
    the official competition macro-averaged F_0.5 score.
    """
    s1_cand_probs: Dict[str, List[Tuple[str, float]]] = {}
    
    for s1_id in val_s1_ids:
        cands = s1_to_candidates.get(s1_id, [])
        if not cands:
            s1_cand_probs[s1_id] = []
            continue
        cand_ids = [c[0] for c in cands]
        X = np.array([c[1] for c in cands])
        probs = clf.predict_proba(X)[:, 1]
        s1_cand_probs[s1_id] = list(zip(cand_ids, probs))

    best_thresh = 0.5
    best_f05 = -1.0
    history = {}

    n_val = len(val_s1_ids)

    for thresh in threshold_range:
        total_f05 = 0.0
        for s1_id in val_s1_ids:
            gt_set = gt_mapping.get(s1_id, set())
            cands = s1_cand_probs.get(s1_id, [])
            pred_set = {cid for cid, p in cands if p >= thresh}
            
            total_f05 += compute_entity_f05(gt_set, pred_set)
        
        macro_score = total_f05 / n_val if n_val > 0 else 0.0
        history[float(thresh)] = macro_score
        
        if macro_score > best_f05:
            best_f05 = macro_score
            best_thresh = float(thresh)

    return best_thresh, best_f05, history


def find_optimal_threshold_two_phase(
    clf: lgb.LGBMClassifier,
    val_s1_ids: List[str],
    s1_to_candidates: Dict[str, List[Tuple[str, np.ndarray]]],
    gt_mapping: Dict[str, Set[str]],
    coarse_range: np.ndarray = np.arange(0.35, 0.90, 0.05),
    fine_radius: float = 0.04,
    fine_step: float = 0.005
) -> Tuple[float, float, Dict[float, float]]:
    """
    Two-Phase Threshold Search:
    Phase 1 (Coarse): Broad scan over coarse_range (e.g. 0.35 to 0.85 by 0.05)
    Phase 2 (Fine): High-precision local zoom around the coarse peak (radius +- 0.04 by 0.005)
    Guarantees pinpoint optimization of macro F_0.5 without expensive dense global sweeps.
    """
    print("  Phase 1: Coarse threshold sweep across range [%.2f, %.2f]..." % (coarse_range[0], coarse_range[-1]))
    best_coarse_th, best_coarse_f05, coarse_history = find_optimal_threshold_for_f05(
        clf, val_s1_ids, s1_to_candidates, gt_mapping, threshold_range=coarse_range
    )
    print(f"  Coarse Peak: tau = {best_coarse_th:.2f} (Macro F_0.5 = {best_coarse_f05:.4f})")

    # Fine search around peak
    fine_min = max(0.20, best_coarse_th - fine_radius)
    fine_max = min(0.98, best_coarse_th + fine_radius + fine_step / 2.0)
    fine_range = np.arange(fine_min, fine_max, fine_step)

    print(f"  Phase 2: Fine zoom across [{fine_min:.3f}, {fine_max:.3f}] with step {fine_step:.3f}...")
    best_fine_th, best_fine_f05, fine_history = find_optimal_threshold_for_f05(
        clf, val_s1_ids, s1_to_candidates, gt_mapping, threshold_range=fine_range
    )

    combined_history = {**coarse_history, **fine_history}
    return best_fine_th, best_fine_f05, combined_history


def save_matcher_model(clf: lgb.LGBMClassifier, threshold: float, filepath: str):
    """Saves trained model and optimal threshold to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({"model": clf, "threshold": threshold}, f)


def load_matcher_model(filepath: str) -> Tuple[lgb.LGBMClassifier, float]:
    """Loads trained model and threshold from disk."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]


if __name__ == "__main__":
    # Self-test synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_feats = 18
    X = np.random.randn(n_samples, n_feats)
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)

    clf = train_lgbm_model(X[:800], y[:800], X[800:], y[800:])
    val_probs = clf.predict_proba(X[800:])[:, 1]
    print(f"Self-test LightGBM AUC / Mean Prob: {np.mean(val_probs):.4f}")
    assert clf is not None
    print("model.py test PASSED!")

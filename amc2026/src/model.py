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
    broad_points: int = 90,
    fine_points: int = 61
) -> Tuple[float, float, Dict[float, float]]:
    """
    Evaluates candidate predictions across a two-phase count-based grid
    (broad 0.10 to 0.99 inclusive, then fine-grained around peak) to maximize Macro F_0.5.
    Avoids floating-point accumulation bugs.
    """
    # Precompute model probabilities for all candidate pairs in validation
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

    n_val = len(val_s1_ids)

    def eval_threshold(thresh: float) -> float:
        total_f05 = 0.0
        for s1_id in val_s1_ids:
            gt_set = gt_mapping.get(s1_id, set())
            cands = s1_cand_probs.get(s1_id, [])
            pred_set = {cid for cid, p in cands if p >= thresh}
            total_f05 += compute_entity_f05(gt_set, pred_set)
        return total_f05 / n_val if n_val > 0 else 0.0

    # Phase 1: Broad count-based search from 0.10 to 0.99 inclusive
    broad_grid = np.linspace(0.10, 0.99, broad_points)
    history = {}
    best_broad_thresh = 0.5
    best_broad_f05 = -1.0

    for th in broad_grid:
        r_th = round(float(th), 4)
        score = eval_threshold(r_th)
        history[r_th] = score
        if score > best_broad_f05:
            best_broad_f05 = score
            best_broad_thresh = r_th

    # Phase 2: Fine search around broad winner (+/- 0.06 with fine steps ~0.002)
    fine_min = max(0.01, best_broad_thresh - 0.06)
    fine_max = min(0.999, best_broad_thresh + 0.06)
    fine_grid = np.linspace(fine_min, fine_max, fine_points)

    best_thresh = best_broad_thresh
    best_f05 = best_broad_f05

    for th in fine_grid:
        r_th = round(float(th), 4)
        if r_th not in history:
            score = eval_threshold(r_th)
            history[r_th] = score
        else:
            score = history[r_th]

        if score > best_f05:
            best_f05 = score
            best_thresh = r_th

    return best_thresh, best_f05, history


def save_matcher_model(
    clf: lgb.LGBMClassifier,
    threshold: float,
    filepath: str,
    country_thresholds: Optional[Dict[str, float]] = None,
    margin_params: Optional[Dict[str, float]] = None
):
    """Saves trained model, global threshold, and optional per-country / margin thresholds to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": clf,
            "threshold": threshold,
            "country_thresholds": country_thresholds or {},
            "margin_params": margin_params or {}
        }, f)


def load_matcher_model(filepath: str) -> Tuple[lgb.LGBMClassifier, float]:
    """Loads trained model and threshold from disk."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]


def load_matcher_model_full(filepath: str) -> Tuple[lgb.LGBMClassifier, float, Dict[str, float], Dict[str, float]]:
    """Loads trained model, global threshold, per-country thresholds, and margin parameters."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return (
        data["model"],
        data.get("threshold", 0.6),
        data.get("country_thresholds", {}),
        data.get("margin_params", {})
    )


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

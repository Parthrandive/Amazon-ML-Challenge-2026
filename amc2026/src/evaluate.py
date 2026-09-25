"""
evaluate.py
Official Competition Metric Evaluation for Amazon ML Challenge 2026.
Macro-averaged F_0.5 score across all Source 1 entities with singleton support.
"""

from typing import Dict, Set, Iterable
import pandas as pd


def compute_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singletons:
    - If gt is empty and pred is empty: 1.0 (correctly identified singleton)
    - If gt is empty and pred is non-empty: 0.0 (false merge on singleton)
    - If gt is non-empty and pred is empty: 0.0 (missed match)
    """
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    hits = len(gt_ids & pred_ids)
    if hits == 0:
        return 0.0

    prec = hits / n_pred
    rec = hits / n_gt

    denom = 0.25 * prec + rec
    if denom == 0:
        return 0.0

    return (1.25 * prec * rec) / denom


def evaluate_predictions(
    gt_mapping: Dict[str, Set[str]],
    pred_mapping: Dict[str, Set[str]],
    required_s1_ids: Iterable[str] = None
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth mapping.
    
    Parameters:
    -----------
    gt_mapping: dict mapping source1_entity_id -> set of true matched_entity_ids
    pred_mapping: dict mapping source1_entity_id -> set of predicted matched_entity_ids
    required_s1_ids: optional list/set of all S1 IDs. If provided, missing entities in pred are treated as empty.
    
    Returns:
    --------
    dict with metrics:
        'macro_f05': overall macro-averaged F_0.5
        'mean_precision': average precision across non-singleton GT entities
        'mean_recall': average recall across non-singleton GT entities
        'singleton_accuracy': accuracy on singleton entities
        'n_total': total entities evaluated
        'n_singletons': number of singleton entities
        'n_with_matches': number of entities with true matches
    """
    if required_s1_ids is None:
        eval_ids = set(gt_mapping.keys()) | set(pred_mapping.keys())
    else:
        eval_ids = set(required_s1_ids)

    total_f05 = 0.0
    singleton_correct = 0
    n_singletons = 0
    n_with_matches = 0
    precisions = []
    recalls = []

    for s1_id in eval_ids:
        gt_set = gt_mapping.get(s1_id, set())
        pred_set = pred_mapping.get(s1_id, set())

        score = compute_entity_f05(gt_set, pred_set)
        total_f05 += score

        if len(gt_set) == 0:
            n_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            n_with_matches += 1
            if len(pred_set) > 0:
                hits = len(gt_set & pred_set)
                precisions.append(hits / len(pred_set))
                recalls.append(hits / len(gt_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)

    n_total = len(eval_ids)
    return {
        "macro_f05": total_f05 / n_total if n_total > 0 else 0.0,
        "mean_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "singleton_accuracy": singleton_correct / n_singletons if n_singletons > 0 else 0.0,
        "n_total": n_total,
        "n_singletons": n_singletons,
        "n_with_matches": n_with_matches,
    }


def load_mapping_from_tsv(path: str, col_name: str) -> Dict[str, Set[str]]:
    """
    Loads TSV into a mapping of source1_entity_id -> set of entity IDs.
    Handles empty strings, nan, and whitespace.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = row.get(col_name)
        if pd.isna(val) or not str(val).strip():
            mapping[s1] = set()
        else:
            mapping[s1] = {x.strip() for x in str(val).split(",") if x.strip()}
    return mapping


if __name__ == "__main__":
    # Self-test using problem statement example
    test_gt = {"S1-00001": {"S2-00047", "S3-00812"}}
    test_pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    f05 = compute_entity_f05(test_gt["S1-00001"], test_pred["S1-00001"])
    print(f"Self-test S1-00001: computed F_0.5 = {f05:.4f}, expected ~0.7143")
    assert abs(f05 - 0.7142857) < 1e-4, "Mismatch in self-test!"
    print("evaluate.py test PASSED!")

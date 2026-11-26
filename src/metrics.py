import re
from typing import List
from seqeval.metrics import f1_score, precision_score, recall_score


def hard_processing(text):
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def comp_metric(y_true: List[List[str]], y_pred: List[List[str]]):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p,  = 0.0, 0.0, 0.0, 0.0

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([hard_processing(s) for s in y_true_sample])
        y_pred_sample = set([hard_processing(s) for s in y_pred_sample])

        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)

    if tp + fp == 0:
        if p == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp/(tp + fp)

    if tp + fn == 0:
        if p == 0:
            recall = 1.0
        else:
            recall = 0.0
    else:
        recall = tp/(tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 8)

def compare_locations(gold, predicted):
    gold = {hard_processing(x) for x in gold}
    predicted = {hard_processing(x) for x in predicted}

    return {
        "true_positives": gold & predicted,
        "false_positives": predicted - gold,
        "false_negatives": gold - predicted
    }

def compute_kaggle_metrics(y_true, y_pred):
    tp, fp, fn = 0, 0, 0

    for gold, predicted in zip(y_true, y_pred):
        comparison = compare_locations(gold, predicted)

        tp += len(comparison["true_positives"])
        fp += len(comparison["false_positives"])
        fn += len(comparison["false_negatives"])

    precision = tp / (tp + fp) if tp + fp else float(tp + fn == 0)
    recall = tp / (tp + fn) if tp + fn else 1.0

    return {
        "f1": comp_metric(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def compute_kaggle_metrics_by_source(y_true, y_pred, sources):
    results = {"combined": compute_kaggle_metrics(y_true, y_pred)}
    for source in sorted(set(sources)):
        indices = [i for i, s in enumerate(sources) if s == source]

        results[source] = compute_kaggle_metrics([y_true[i] for i in indices], [y_pred[i] for i in indices])
    return results


def compute_bio_metrics(y_true, y_pred):
    return {
        "entity_f1": f1_score(y_true, y_pred, zero_division=0),
        "entity_precision": precision_score(y_true, y_pred, zero_division=0),
        "entity_recall": recall_score(y_true, y_pred, zero_division=0)
    }
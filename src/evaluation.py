import ast
import torch

from src.decoding import viterbi_decode, decode_locations
from src.metrics import compute_bio_metrics, compute_kaggle_metrics, compute_kaggle_metrics_by_source


def decode_batch(logits, rows, label_to_id):
    emissions, records, lengths = [], [], []
    token_indices = []

    for i, row in enumerate(rows):
        indices = [j for j, (a, b) in enumerate(row["offset_mapping"]) if a < b]
        offsets = [row["offset_mapping"][j] for j in indices]
        token_indices.append(indices)
        emissions.append(logits[i, indices])
        records.append({"text": row["text"], "offsets": offsets})
        lengths.append(len(indices))

    if max(lengths) == 0:
        return [[] for _ in rows], token_indices

    padded = torch.nn.utils.rnn.pad_sequence(emissions, batch_first=True)
    predictions = viterbi_decode(padded, lengths, label_to_id, records)
    return predictions, token_indices


def build_compute_metrics(dataset, label_to_id):
    id_to_label = {v: k for k, v in label_to_id.items()}
    cursor = 0
    true_bio, pred_bio = [], []
    true_locations, pred_locations = [], []
    sources = []

    def compute_metrics(eval_pred, compute_result=False):
        nonlocal cursor

        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs

        batch_size = len(logits)
        rows = [dataset[i] for i in range(cursor, cursor + batch_size)]

        for i, row in enumerate(rows):
            actual_ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
            if actual_ids != row["input_ids"]:
                raise ValueError("Evaluation dataset order mismatch")

        predictions, token_indices = decode_batch(logits, rows, label_to_id)

        for row, pred, indices, gold in zip(rows, predictions, token_indices, labels):
            offsets = [row["offset_mapping"][i] for i in indices]

            true_bio.append([id_to_label[int(gold[i])] for i in indices if gold[i] != -100])
            pred_bio.append([id_to_label[int(p)] for i, p in zip(indices, pred) if gold[i] != -100])

            if "locations" in row:
                locations = row["locations"]
                if isinstance(locations, str):
                    locations = ast.literal_eval(locations)
            else:
                spans = ast.literal_eval(row["loc_markers"])
                locations = [row["text"][start:end] for start, end in spans]

            true_locations.append(locations)
            pred_locations.append(decode_locations(pred, offsets, row["text"], id_to_label))

            if "source" in row:
                sources.append(row["source"])

        cursor += batch_size

        if not compute_result:
            return {}

        metrics = compute_bio_metrics(true_bio, pred_bio)
        if sources:
            scores = compute_kaggle_metrics_by_source(true_locations, pred_locations, sources)
            metrics.update(scores["combined"])

            for source, values in scores.items():
                if source != "combined":
                    metrics.update({f"{source}_{k}": v for k, v in values.items()})
        else:
            metrics.update(compute_kaggle_metrics(true_locations, pred_locations))

        cursor = 0
        for values in (true_bio, pred_bio, true_locations, pred_locations, sources):
            values.clear()
        return metrics

    return compute_metrics
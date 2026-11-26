import numpy as np
import pandas as pd
from collections import Counter
from src.eda_scan import has_pair_shape
from src.eda_scan import parse_spans

def drop_conflicting_annotations(df):
    conflicts = df.groupby("text")["loc_markers"].transform("nunique") > 1
    return df.loc[~conflicts].copy()


def drop_duplicate_texts(df):
    normalized = df["text"].str.lower().str.split().str.join(" ")
    return df.loc[~normalized.duplicated()].copy()


def preprocess_data(df):
    df = drop_conflicting_annotations(df)
    df = drop_duplicate_texts(df)
    return df


def spans_to_bio(offsets, spans, label_to_id, strict=True):
    labels = [label_to_id["O"] if a < b else -100 for a, b in offsets]

    for start, end in spans:
        indices = [i for i, (a, b) in enumerate(offsets) if a < b and a < end and b > start]

        if not indices:
            raise ValueError(f"No tokens for span {(start, end)}")

        if strict and (offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end):
            raise ValueError(f"Partial alignment for span {(start, end)}")

        labels[indices[0]] = label_to_id["B-LOC"]
        for i in indices[1:]:
            labels[i] = label_to_id["I-LOC"]

    return labels


def tokenize_batch(batch, tokenizer, label_to_id, max_length, strict=True):
    encoding = tokenizer(batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
    labels = []
    valid = []

    for offsets, raw in zip(encoding["offset_mapping"], batch["loc_markers"]):
        spans = parse_spans(raw)
        try:
            if not has_pair_shape(spans):
                raise ValueError("Invalid spans")
            labels.append(spans_to_bio(offsets, spans, label_to_id, strict=strict))
            valid.append(True)
        except ValueError:
            labels.append([-100] * len(offsets))
            valid.append(False)

    encoding["labels"] = labels
    encoding["valid"] = valid
    return encoding


def bio_to_spans(labels, offsets):
    if len(labels) != len(offsets):
        raise ValueError("Labels and offsets must have equal lengths")
    spans = []
    start, end = None, None
    for label, (a, b) in zip(labels, offsets):
        if label == "B-LOC" or (label == "I-LOC" and start is None):
            if start is not None:
                spans.append((start, end))
            start, end = a, b
        elif label == "I-LOC":
            end = max(end, b)
        elif label == "O":
            if start is not None:
                spans.append((start, end))
                start, end = None, None
        else:
            raise ValueError(f"Unknown BIO label: {label}")
    if start is not None:
        spans.append((start, end))
    return spans


def serialize_spans(spans):
    return spans if isinstance(spans, str) else repr([tuple(map(int, span)) for span in spans])


def evaluate_tokenizer(tokenizer, datasets, max_length=512):
    results = {}

    for name, df in datasets.items():
        lengths = []
        unknown = 0
        alignment = Counter()

        for row in df.itertuples():
            encoding = tokenizer(row.text, return_offsets_mapping=True, truncation=False, verbose=False)
            ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            lengths.append(len(ids))
            unknown += ids.count(tokenizer.unk_token_id)

            if hasattr(row, "loc_markers"):
                spans = parse_spans(row.loc_markers)
                if not has_pair_shape(spans):
                    continue

                for start, end in spans:
                    overlapping = [(a, b) for a, b in offsets if a < b and a < end and b > start]

                    if not overlapping:
                        alignment["missing"] += 1
                    elif overlapping[0][0] == start and overlapping[-1][1] == end:
                        alignment["exact"] += 1
                    else:
                        alignment["partial"] += 1

        total_spans = sum(alignment.values())
        results[name] = {
            "Texts": len(lengths),
            "Median tokens": np.median(lengths),
            "P95 tokens": np.percentile(lengths, 95),
            "Max tokens": max(lengths),
            "Over max length (%)": 100 * np.mean(np.array(lengths) > max_length),
            "Unknown tokens (%)": 100 * unknown / sum(lengths),
            "Exact alignment (%)": 100 * alignment["exact"] / total_spans if total_spans else np.nan,
            "Partial alignment (%)": 100 * alignment["partial"] / total_spans if total_spans else np.nan,
            "Missing alignment (%)": 100 * alignment["missing"] / total_spans if total_spans else np.nan
        }

    return pd.DataFrame(results).round(3)
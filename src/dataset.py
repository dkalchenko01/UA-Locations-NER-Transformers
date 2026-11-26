import pandas as pd
from datasets import Dataset, DatasetDict
from src.preprocessing import tokenize_batch
from pathlib import Path
from datasets import load_from_disk


def create_datasets(splits, sources):
    datasets = DatasetDict()

    for split in ("train", "val", "holdout"):
        df = pd.concat([splits[f"{source}_{split}"].assign(source=source) for source in sources])
        df = df.rename_axis("row_id").reset_index()
        df["text_id"] = df["source"] + "_" + df["row_id"].astype(str)
        datasets[split] = Dataset.from_pandas(df, preserve_index=False)

    return datasets


def tokenize_datasets(datasets, tokenizer, label_to_id, max_length):
    result = DatasetDict()

    for split, dataset in datasets.items():
        tokenized = dataset.map(
            tokenize_batch, batched=True,
            fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id,
                       "max_length": max_length, "strict": split == "train"}
        )

        print(f"{split}: {tokenized['valid'].count(False):,} alignment failures")

        if split == "train":
            tokenized = tokenized.filter(lambda row: row["valid"])

        result[split] = tokenized

    return result

def prepare_labeling_dataset(df, tokenizer, label_to_id, max_length):
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.map(
        tokenize_batch, batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_to_id": label_to_id,
            "max_length": max_length,
            "strict": False})

def load_or_tokenize_datasets(datasets, tokenizer, label_to_id, max_length, cache_path):
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading tokenized datasets: {cache_path.name}")
        return load_from_disk(str(cache_path))

    tokenized = tokenize_datasets(datasets, tokenizer, label_to_id, max_length)
    tokenized.save_to_disk(str(cache_path))
    return tokenized
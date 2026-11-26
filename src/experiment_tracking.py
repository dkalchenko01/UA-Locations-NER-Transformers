from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import wandb

DEFAULT_ENTITY = "daldoshyna-kyiv-school-of-economics"
DEFAULT_PROJECT = "ua-location-ner-transformers"

def sha256_file(path: str | Path) -> str:
    """Fingerprint an existing file without uploading its content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(repo_dir: str | Path) -> dict[str, Any]:
    """Read local Git commit and whether tracked/untracked files changed."""
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def start_run(
    *,
    model: str,
    config: Mapping[str, Any],
    name: str | None = None,
    validation_version: str = "v2",
    metric_version: str = "kaggle_original_regex_v1",
    postprocessing_version: str = "raw",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    repo_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
    val_manifest_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
    job_type: str = "train",
) -> Any:
    """Create a W&B run with comparable config and reproducibility metadata.

    Pass the existing notebook's PROVENANCE if available. File paths here are
    hashed locally; files are NOT uploaded. No credentials are logged.
    """
    if not model.strip():
        raise ValueError("model must be a non-empty name")

    metadata: dict[str, Any] = {
        "model": model,
        "validation_version": validation_version,
        "metric_version": metric_version,
        "postprocessing_version": postprocessing_version,
    }
    if repo_dir is not None:
        metadata.update(_git_info(repo_dir))
    if split_manifest_path is not None:
        metadata["split_manifest_sha256"] = sha256_file(split_manifest_path)
    if train_manifest_path is not None:
        metadata["train_manifest_sha256"] = sha256_file(train_manifest_path)
    if val_manifest_path is not None:
        metadata["val_manifest_sha256"] = sha256_file(val_manifest_path)
    if tokenizer_path is not None:
        metadata["tokenizer_sha256"] = sha256_file(tokenizer_path)

    if provenance is not None:
        metadata["provenance"] = dict(provenance)
        metadata.setdefault("git_commit", provenance.get("git_revision"))
        metadata.setdefault("tokenizer_sha256", provenance.get("tokenizer_sha256"))
        metadata.setdefault("split_manifest_sha256", provenance.get("manifest_sha256"))
        metadata["run_id"] = provenance.get("run_id", name)

    # Explicit metadata wins over any coincidentally named model hyperparameter.
    run_config = {**dict(config), **metadata}
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        group=model.lower(),
        job_type=job_type,
        tags=list(tags),
        config=run_config,
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")
    
    # Use Trainer's global step as the W&B history x-axis
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")
    return run



def log_evaluation(
    run: Any,
    dataset: str,
    metrics: Mapping[str, int | float],
    *,
    postprocessing_version: str = "raw",
    kind: str = "diagnostic",
    examples: Sequence[Mapping[str, Any]] | None = None,
    upload_examples: bool = False,
    include_text: bool = False,
    max_examples: int = 30,
) -> None:
    """Store final scores, e.g. labeling_sample raw/cleaned F1.

    Examples are NOT uploaded unless upload_examples=True. Even then, text is
    excluded unless include_text=True. Check dataset permissions first.
    """
    if not dataset or not postprocessing_version:
        raise ValueError("dataset and postprocessing_version are required")
    prefix = f"{kind}/{dataset}/{postprocessing_version}"
    for metric_name, value in metrics.items():
        if value is not None:
            run.summary[f"{prefix}/{metric_name}"] = float(value)

    if upload_examples and examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        columns = ["text_id", "gold", "predicted", "false_positives", "false_negatives"]
        if include_text:
            columns.insert(1, "text")
        table = wandb.Table(columns=columns)
        for example in examples[:max_examples]:
            values = []
            for column in columns:
                value = example.get(column)
                if value is None and column == "gold":
                    value = example.get("gold_locations")
                if value is None and column == "predicted":
                    value = example.get("locations")
                values.append(str(value) if value is not None else "")
            table.add_data(*values)
        run.log({f"{prefix}/examples": table})


def log_checkpoint(run, path, *, artifact_name=None, metadata=None):
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = str(run.config.get("model", "model")).lower()
    default_name = f"{model}-{run.id}-best"
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", artifact_name or default_name)
    artifact = wandb.Artifact(name=name, type="model", metadata=dict(metadata or {}))

    if checkpoint.is_dir():
        artifact.add_dir(str(checkpoint))
    else:
        artifact.add_file(str(checkpoint))

    run.log_artifact(artifact)
    run.summary["best/checkpoint_artifact"] = name
    run.summary["best/checkpoint_local"] = str(checkpoint)


def finish_run(
    run: Any,
    *,
    summary: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    best_score: float | None = None,
    selection_metric: str = "f1",
    checkpoint_path: str | Path | None = None,
    upload_checkpoint: bool = False,
) -> None:
    """Set summary, optionally upload checkpoint, then close the run."""
    if summary is not None:
        for key, value in summary.items():
            if value is not None:
                run.summary[key] = value
    if best_epoch is not None:
        run.summary["best/epoch"] = int(best_epoch)
    if best_score is not None:
        run.summary[f"best/{selection_metric}"] = float(best_score)
        run.summary["best/selection_metric"] = selection_metric
    if checkpoint_path is not None:
        run.summary["best/checkpoint_local"] = str(checkpoint_path)
        if upload_checkpoint:
            log_checkpoint(run, checkpoint_path)
    elif upload_checkpoint:
        raise ValueError("Provide checkpoint_path when upload_checkpoint=True")
    run.finish()


def log_kaggle_result(
    run_or_path: Any,
    *,
    public_f1: float,
    postprocessing_version: str,
    submission_filename: str,
) -> None:
    """Record a public score, even after a run has finished.

    run_or_path: active wandb Run OR 'entity/project/run_id' string.
    Public Kaggle F1 is reporting-only; never a Sweep optimization target.
    """
    if not 0.0 <= float(public_f1) <= 1.0:
        raise ValueError("public_f1 must be between 0 and 1")
    key = f"kaggle/{postprocessing_version}"
    if isinstance(run_or_path, str):
        remote_run = wandb.Api().run(run_or_path)
        remote_run.summary[f"{key}/public_f1"] = float(public_f1)
        remote_run.summary[f"{key}/submission_filename"] = submission_filename
        remote_run.summary.update()
    else:
        run_or_path.summary[f"{key}/public_f1"] = float(public_f1)
        run_or_path.summary[f"{key}/submission_filename"] = submission_filename


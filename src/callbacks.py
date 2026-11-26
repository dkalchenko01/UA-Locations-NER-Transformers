import shutil
from pathlib import Path
from transformers import TrainerCallback
from src.evaluation import build_compute_metrics


class CheckpointBackupCallback(TrainerCallback):
    def __init__(self, backup_dir):
        self.backup_dir = Path(backup_dir)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        destination = self.backup_dir / checkpoint.name
        shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
        print(f"Checkpoint backed up: {destination}")

class LabelingSampleCallback(TrainerCallback):
    def __init__(self, trainer, dataset, label_to_id):
        self.trainer = trainer
        self.dataset = dataset
        self.label_to_id = label_to_id

    def on_evaluate(self, args, state, control, **kwargs):
        original_metrics = self.trainer.compute_metrics
        try:
            self.trainer.compute_metrics = build_compute_metrics(self.dataset, self.label_to_id)
            result = self.trainer.predict(self.dataset, metric_key_prefix="labeling")
            self.trainer.log(result.metrics)
        finally:
            self.trainer.compute_metrics = original_metrics
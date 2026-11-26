import torch


def build_optimizer(trainer, config):
    model = trainer.model
    decay_names = set(trainer.get_decay_parameter_names(model))
    groups = {}

    num_layers = len(model.encoder.encoder.layer)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder.encoder.layer."):
            layer_id = int(name.split(".")[3])
            depth = num_layers - 1 - layer_id
            lr = config["encoder_lr"] * config["layer_lr_decay"] ** depth
        else:
            lr = config["head_lr"]

        wd = config["weight_decay"] if name in decay_names else 0.0

        groups.setdefault((lr, wd), []).append(param)

    return torch.optim.AdamW([
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in groups.items()
    ])
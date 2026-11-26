import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states):
        layers = torch.stack(hidden_states[-self.num_layers:])
        weights = torch.softmax(self.weights, dim=0).to(layers.dtype)
        return (layers * weights[:, None, None, None]).sum(dim=0)


class TransformerNER(nn.Module):
    def __init__(self, model_name, num_labels=3, pooling_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = WeightedLayerPooling(pooling_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden = self.pooling(outputs.hidden_states)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)

    def freeze_encoder(self, trainable_layers):
        layers = self.encoder.encoder.layer

        if not 0 <= trainable_layers <= len(layers):
            raise ValueError("Invalid number of trainable layers")

        self.encoder.requires_grad_(False)
        if trainable_layers > 0:
            for layer in layers[-trainable_layers:]:
                layer.requires_grad_(True)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()
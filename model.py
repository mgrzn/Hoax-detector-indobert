import torch
import torch.nn as nn
from transformers import AutoModel

class IndoBERTHoaxClassifier(nn.Module):
    def __init__(self, model_name="indobenchmark/indobert-base-p1", freeze_layers=8, num_labels=2):
        super(IndoBERTHoaxClassifier, self).__init__()

        # Load pretrained IndoBERT
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze beberapa layer awal supaya training lebih stabil
        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Dropout + classifier
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.nn.functional.softmax(logits, dim=1) if not self.training else logits


import torch
import torch.nn as nn
from transformers import BertModel

class DoubleInputBERTModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 6):
        super(DoubleInputBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        labels=None
    ):
        # Branch 1: Statement
        outputs1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Branch 2: Justification
        outputs2 = self.bert(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2
        )

        # Use [CLS] token embedding (first position)
        pooled1 = outputs1.last_hidden_state[:, 0]  # (batch_size, hidden_size)
        pooled2 = outputs2.last_hidden_state[:, 0]

        # Concatenate and classify
        concat = torch.cat((pooled1, pooled2), dim=1)  # (batch_size, hidden_size * 2)
        logits = self.classifier(self.dropout(concat))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

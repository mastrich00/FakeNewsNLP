import torch
import torch.nn as nn
from transformers import RobertaModel

class DoubleInputRoBERTaModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 6):
        super(DoubleInputRoBERTaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_2=None,
        attention_mask_2=None,
        labels=None
    ):
        # RoBERTa does NOT use token_type_ids
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs2 = self.roberta(input_ids=input_ids_2, attention_mask=attention_mask_2)

        # Use the first token ([CLS]-equivalent in RoBERTa) from both branches
        pooled1 = outputs1.last_hidden_state[:, 0]  # shape: (batch_size, hidden_size)
        pooled2 = outputs2.last_hidden_state[:, 0]

        # Concatenate and classify
        concat = torch.cat((pooled1, pooled2), dim=1)  # shape: (batch_size, hidden_size * 2)
        logits = self.classifier(self.dropout(concat))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

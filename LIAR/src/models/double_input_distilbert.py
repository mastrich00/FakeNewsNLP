import torch
import torch.nn as nn
from transformers import AutoModel

class DoubleInputBERTModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=6):
        super(DoubleInputBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)  # *2 for concatenated output

    def forward(self, input_ids=None, attention_mask=None,
                input_ids_2=None, attention_mask_2=None, labels=None):
        # Encode first input (statement)
        outputs_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output_1 = outputs_1.last_hidden_state[:, 0]  # CLS token

        # Encode second input (justification)
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        pooled_output_2 = outputs_2.last_hidden_state[:, 0]  # CLS token

        # Concatenate
        combined = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits}

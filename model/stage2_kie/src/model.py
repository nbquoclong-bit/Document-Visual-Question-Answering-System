from typing import Optional

import torch
import torch.nn as nn
from transformers import LayoutLMv3PreTrainedModel, LayoutLMv3Model
from transformers.modeling_outputs import TokenClassifierOutput


class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, bbox=None, labels=None, **kwargs):
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_model(
    model_name: str = "microsoft/layoutlmv3-base",
    num_labels: int = 9,
    id2label: Optional[dict] = None,
    label2id: Optional[dict] = None,
):
    """Load a LayoutLMv3 token-classification model with a configurable number of labels."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModel.from_pretrained(model_name, config=cfg, trust_remote_code=False)
    return model


def push_model_to_hub(model, repo_name: str, token: str):
    """Push a trained model (and optional tokenizer) to the Hugging Face Hub."""
    import huggingface_hub  # noqa: F401 – ensure it is available

    model.push_to_hub(repo_name, token=token)

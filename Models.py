import torch
import torch.nn as nn
from transformers import *
from transformers.modeling_utils import *


class Sci_BERT(nn.Module):
    def __init__(self, num_classes):
        super(Sci_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(
            "allenai/scibert_scivocab_cased",
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask=att_mask)
        output = self.linear(pooled_output)
        return output


class RoBERTa(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTa, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            "roberta-base",
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input, att_mask):
        _, pooled_output, _ = self.bert(input, attention_mask=att_mask)
        output = self.linear(pooled_output)
        return output

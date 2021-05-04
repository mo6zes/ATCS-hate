import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import create_pretrained_transformer

class BERT(nn.Module):
    def __init__(self, transformer_model='bert-base-uncased', hidden_size=512, output_size=512):
        super().__init__()
        
        self.bert = create_pretrained_transformer(transformer_model)
        self.mlp = nn.Sequential(
                      nn.Linear(self.bert.config.hidden_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, output_size),
                    ) if output_size > 0 else nn.Identity()
        
        self.freeze_bert()

    def forward(self, x):
        input, attention_mask = x
        output = self.bert(input, attention_mask=attention_mask)
        # we only take the hidden state of the CLS token.
        output = self.mlp(output.last_hidden_state[:, 0, :])
        return output #[B, C]
    
    def freeze_bert(self):
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import create_pretrained_transformer

class BERT(nn.Module):
    def __init__(self, transformer_model='bert-base-uncased', hidden_size=512,
                 output_size=512, activation=nn.ReLU(), gradient_checkpointing=False):
        super().__init__()

        self.bert = create_pretrained_transformer(transformer_model, gradient_checkpointing=gradient_checkpointing)
        self.freeze_bert()
        self.mlp = nn.Sequential(
                      nn.Linear(self.bert.config.hidden_size, hidden_size),
                      activation,
                      nn.Linear(hidden_size, output_size),
                    ) if output_size > 0 else nn.Identity()
        
    def feature_forward(self, x, attention_mask=None):
        return self.bert(x, attention_mask=attention_mask)

    def forward(self, x):
        input, attention_mask = x
        output = self.feature_forward(input, attention_mask=attention_mask)
        # we only take the hidden state of the CLS token.
        output = self.mlp(output.last_hidden_state[:, 0, :])
        return output  # [B, C]

    def freeze_bert(self, freeze=True):
        """Freeze the entire BERT model."""
        for param in self.bert.base_model.parameters():
            param.requires_grad = False if freeze else True
            
    def grads(self, module, freeze=True):
        """Helper function to freeze or unfreeze part of the model."""
        for i in module.modules():
            for param in i.parameters():
                param.requires_grad = False if freeze else True

    def unfreeze_module(self, module_instance=nn.LayerNorm, freeze: bool=False):
        """Unfreeze a particular module type in BERT."""
        for module in self.bert.base_model.modules():
            if isinstance(module, module_instance):
                for param in module.parameters():
                    param.requires_grad = False if freeze else True
                    
    def unfreeze_attention_layer(self, layer_numbers: list=None, pooler: bool=False, freeze: bool=False):
        """Unfreeze a particular attention layer (0-11).
           Also can unfreeze the final pooling layer."""
        if pooler:
            self.grads(self.bert.base_model.pooler, freeze=freeze)

        if layer_numbers is not None:
            for i in layer_numbers:
                self.grads(self.bert.base_model.encoder.layer[i], freeze=freeze)
                        
    def reset_mlp(self, hidden_size: int, output_size: int, activation=nn.ReLU()):
        self.mlp = nn.Sequential(
                      nn.Linear(self.bert.config.hidden_size, hidden_size),
                      activation,
                      nn.Linear(hidden_size, output_size),
                    ) if output_size > 0 else nn.Identity()

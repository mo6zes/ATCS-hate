from transformers import AutoModel, AutoTokenizer


def create_pretrained_transformer(model_type: str='bert-base-uncased', gradient_checkpointing: bool=False):
    model = AutoModel.from_pretrained(model_type, gradient_checkpointing=gradient_checkpointing)
    return model

def create_tokenizer(model_type: str='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    return tokenizer
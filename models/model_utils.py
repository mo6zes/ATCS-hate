from transformers import AutoModel, AutoTokenizer


def create_pretrained_transformer(model_type: str='bert-base-uncased'):
    model = AutoModel.from_pretrained(model_type)
    return model

def create_tokenizer(model_type: str='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    return tokenizer
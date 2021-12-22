from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name):
    return AutoModel.from_pretrained(model_name)


def _mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return pooled


def encode(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def chunks(l, n):
    """Yield successive n-sized chunks from lst."""
    chunked = []
    for i in range(0, len(l), n):
        chunked.append(l[i : i + n])

    return chunked

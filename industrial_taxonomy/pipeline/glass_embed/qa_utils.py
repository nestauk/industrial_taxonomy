import numpy as np
import matplotlib.pyplot as plt


def full_text_tokenize(texts, tokenizer):
    """Tokenize texts without truncation"""
    tokenized = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    return tokenized


def tokenized_original_length(tokenized):
    """Get number of tokens in original text after tokenization."""
    return np.array(tokenized["attention_mask"].sum(axis=1))


def tokenized_length_histogram(lengths, tokenizer):
    """Plots a cumulative histogram of tokenized text lengths with vertical
    line indicating `model_max_length`.
    """
    max_tokens = np.max(lengths)
    model_max_length = tokenizer.model_max_length

    fig, ax = plt.subplots()
    ax.hist(
        lengths,
        cumulative=True,
        histtype="step",
        density="normed",
        bins=max_tokens,
    )
    ax.axvline(model_max_length, color="gray", linestyle="--")
    ax.set_xlim(0, max_tokens)
    ax.set_xlabel("N Tokens")
    ax.set_ylabel("Cumulative Frequency (norm)")

    return fig

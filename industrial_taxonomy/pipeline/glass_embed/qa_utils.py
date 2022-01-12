import numpy as np
import matplotlib.pyplot as plt


def _tokenize_full_text(texts, tokenizer):
    """Tokenize texts without truncation."""
    return tokenizer(texts, padding=True, truncation=False, return_tensors="np")


def token_count(texts, tokenizer):
    """Get number of tokens in original text after tokenization."""
    tokenized = _tokenize_full_text(texts, tokenizer)
    return np.array(tokenized["attention_mask"].sum(axis=1))


def tokenized_length_histogram(lengths, tokenizer):
    """Plots a cumulative histogram of tokenized text lengths with vertical
    line indicating `model_max_length`.

    Args:
        lengths: List of lengths (number of tokens) of tokenized texts
        tokenizer: A `transformers` tokenizer

    Returns:
        fig: Cumulative normalised histogram of document lengths with a
            vertical line indicating the max tokens of the model
            associated with `tokenizer`.
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
    ax.set_xlabel("N Tokens")
    ax.set_xlim(0, max_tokens)
    ax.set_ylabel("Cumulative Frequency (norm)")

    return fig


def random_sample_idx(x, n_samples):
    """Get a random sample of elements from a list-like without replacement."""
    n_total = len(x)
    rn_gen = np.random.default_rng()
    sample_idx = rn_gen.choice(n_total, size=n_samples, replace=False)
    return sample_idx

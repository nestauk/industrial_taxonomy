"""Utility functions for generating embeddings from Glass descriptions"""


def chunks(l, n):
    """Yield successive n-sized chunks from l.

    Return chunks as a list so that they can be passed into `foreach` flow step.
    """
    chunked = []
    for i in range(0, len(l), n):
        chunked.append(l[i : i + n])
    return chunked

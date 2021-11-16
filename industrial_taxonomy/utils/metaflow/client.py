"""Metaflow utilities relating to the Client API.

https://docs.metaflow.org/metaflow/client
"""
from contextlib import contextmanager

from metaflow import get_namespace, namespace


@contextmanager
def namespace_context(ns: str) -> str:
    """Context manager to temporarily enter metaflow namespace `ns`."""
    old_ns = get_namespace()
    namespace(ns)
    yield ns
    namespace(old_ns)

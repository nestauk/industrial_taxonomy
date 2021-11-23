"""Metaflow utilities relating to the Client API.

https://docs.metaflow.org/metaflow/client
"""
from contextlib import contextmanager
from typing import Generator, Optional

from metaflow import get_namespace, namespace


@contextmanager
def namespace_context(ns: Optional[str]) -> Generator[Optional[str], None, None]:
    """Context manager to temporarily enter metaflow namespace `ns`."""
    old_ns = get_namespace()
    namespace(ns)
    yield ns
    namespace(old_ns)

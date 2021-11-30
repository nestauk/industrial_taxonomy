"""Metaflow utilities relating to the Client API.

https://docs.metaflow.org/metaflow/client
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Optional

from metaflow import Flow, get_namespace, namespace, Run
from metaflow.exception import MetaflowNotFound


@contextmanager
def namespace_context(ns: Optional[str]) -> Generator[Optional[str], None, None]:
    """Context manager to temporarily enter metaflow namespace `ns`."""
    old_ns = get_namespace()
    namespace(ns)
    yield ns
    namespace(old_ns)


@lru_cache()
def get_run(flow_name: str) -> Run:
    """Last successful run of `flow_name` executed with `--production`."""
    runs = Flow(flow_name).runs("project_branch:prod")
    try:
        return next(filter(lambda run: run.successful, runs))
    except StopIteration as exc:
        raise MetaflowNotFound("Matching run not found") from exc

"""Data getters for SIC taxonomy."""
from typing import Dict, Optional

from metaflow import Run

from industrial_taxonomy.pipeline.sic.utils import LEVELS
from industrial_taxonomy.utils.metaflow import get_run

# Type aliases
code_name = str
code = str


def level_lookup(level: int, run: Optional[Run] = None) -> Dict[code, code_name]:
    """Get SIC names for `level` index.

    Args:
        level: Number of SIC digits/letters to fetch lookup for
        run: Metaflow Run to get data artifacts from

    Returns:
        Lookup from SIC code to name

    Raises:
        ValueError: if 1 <= level <= 5
    """
    if run is None:
        run = get_run("Sic2007Structure")

    if level not in range(1, len(LEVELS) + 1):
        raise ValueError(f"Level: {level} not valid.")

    return getattr(run.data, f"{LEVELS[level - 1]}_lookup")

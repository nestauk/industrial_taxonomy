"""Shared utils"""

from typing import Dict


def reverse_dict(_dict: dict) -> Dict:
    """Turns keys into values and viceversa"""
    return {v: k for k, v in _dict.items()}

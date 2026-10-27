"""Canonical collection runtime with isolated state for each loaded tree."""
from ._loader import load_sibling


def load_runtime_tree():
    """Load fresh modules before the prospective collector installs adapters."""
    return load_sibling(__name__, 'clearer_examples')

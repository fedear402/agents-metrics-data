"""Load sibling modules without sharing mutable collection state."""
import importlib.util
from pathlib import Path


def load_sibling(package, name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(package + '.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

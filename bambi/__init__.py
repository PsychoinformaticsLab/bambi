import logging

from pkg_resources import get_distribution

from pymc import math

from .backend import PyMCModel
from .data import clear_data_home, load_data
from .families import Family, Likelihood, Link
from .formula import Formula
from .models import Model
from .priors import Prior
from . import interpret

__version__ = get_distribution("bambi").version

__all__ = [
    "Model",
    "Prior",
    "Family",
    "Likelihood",
    "Link",
    "PyMCModel",
    "Formula",
    "clear_data_home",
    "load_data",
    "math",
]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

#!/usr/bin/env python

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import data, genomics, graph, num, plot, models, utils
from .utils import config, log
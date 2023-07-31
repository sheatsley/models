"""
This module initializes the deep learning models repo.
"""

import pathlib
import subprocess

import dlm.templates as templates
from dlm.models import CNNClassifier, LinearClassifier, MLPClassifier

__all__ = [
    "CNNClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "templates",
]
try:
    __version__ = subprocess.check_output(
        ("git", "-C", *__path__, "rev-parse", "--short", "HEAD"), text=True
    ).strip()
except subprocess.CalledProcessError:
    with open(pathlib.Path(__file__).parent / "VERSION", "r") as f:
        __version__ = f.read().strip()

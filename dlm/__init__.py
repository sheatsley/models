"""
This module initializes the deep learning models repo.
"""

import pathlib
import subprocess

import dlm.pretrained as pretrained
import dlm.templates as templates
import dlm.utilities as utilities
from dlm.models import CNNClassifier, LinearClassifier, MLPClassifier

__all__ = [
    "CNNClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "pretrained",
    "templates",
    "utilities",
]
try:
    cmd = ("git", "-C", *__path__, "rev-parse", "--short", "HEAD")
    __version__ = subprocess.check_output(
        cmd, stderr=subprocess.DEVNULL, text=True
    ).strip()
except subprocess.CalledProcessError:
    with open(pathlib.Path(__file__).parent / "VERSION", "r") as f:
        __version__ = f.read().strip()

"""
This module initializes the deep learning models repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""

import subprocess

import dlm.pretrained as pretrained
import dlm.templates as templates
from dlm.models import CNNClassifier, LinearClassifier, MLPClassifier

__all__ = [
    "CNNClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "pretrained",
    "templates",
]

# compute version
__version__ = subprocess.check_output(
    ("git", "-C", *__path__, "rev-parse", "--short", "HEAD"), text=True
).strip()

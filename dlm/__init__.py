"""
This module initializes the deep learning models repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""

import subprocess

from dlm.models import CNNClassifier, LinearClassifier, MLPClassifier
from dlm.templates import cicmalmem2022, fmnist, mnist, nslkdd, phishing, unswnb15

__all__ = [
    "CNNClassifier",
    "LinearClassifier",
    "MLPClassifier",
    "cicmalmem2022",
    "fmnist",
    "mnist",
    "nslkdd",
    "phishing",
    "unswnb15",
]

# compute version
__version__ = subprocess.check_output(
    ("git", "-C", *__path__, "rev-parse", "--short", "HEAD"), text=True
).strip()

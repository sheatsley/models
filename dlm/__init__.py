"""
This module initializes the deep learning models repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""

# import modules
from dlm.models import LinearClassifier  # base class for all models
from dlm.models import MLPClassifier  # feedforward deep neural network
from dlm.models import CNNClassifier  # convolutional neural network
from dlm.templates import cicmalmem2022  # template for cicmalmem2022
from dlm.templates import fmnist  # template for fmnist
from dlm.templates import nslkdd  # template for nslkdd
from dlm.templates import mnist  # template for mnist
from dlm.templates import phishing  # template for phishing
from dlm.templates import unswnb15  # template for unswnb15
import subprocess  # Subprocess management

# compute version
__version__ = subprocess.check_output(
    ("git", "-C", *__path__, "rev-parse", "--short", "HEAD"), text=True
).strip()

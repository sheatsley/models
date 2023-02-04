"""
This module initializes the deep learning models repo.
Author: Ryan Sheatsley and Blaine Hoak
Tue Nov 29 2022
"""

# import modules
import dlm.architectures as architectures  # hyperparameters and architecture bookkeeping
from dlm.models import LinearClassifier  # base class for all models
from dlm.models import MLPClassifier  # feedforward deep neural network
from dlm.models import CNNClassifier  # convolutional neural network
import dlm.utilities as utilities  # miscellaneous utility functions
import subprocess  # Subprocess management

# compute version
__version__ = subprocess.check_output(
    ("git", "rev-parse", "--short", "HEAD"), text=True
).strip()

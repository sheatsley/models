"""
This module defines interfaces to load and instantiate dlm models from a
variety of pretrained models from popular frameworks, competitions, and papers.
For models that rely on adversarial examples, attacks are provided by
https://github.com/sheatsley/attacks.
"""

import aml
import torch

import dlm


def __getattr__(model):
    """
    This function leverages PEP 562 (https://www.python.org/dev/peps/pep-0562)
    to quickly load pretrained models. At this time, the following models are
    available:

        Madry Lab MNIST Challenge: This loads the secret model used in the
        MNIST adversarial examples challenge
        (https://github.com/MadryLab/mnist_challenge). The model architecture
        is a convolutional neural network with two convolutional layers (with
        max-pooling) and one fully connected layer. The model is ~98% accurate
        on the MNIST test set with a drop of ~10% accuracy against the leading
        white-box attack in the competition. The original hyperparameters are
        not necessary (nor used), but provided for reference.

    :param model: model to load
    :type model: str
    :return: pretrained model
    :rtype: dlm LinearClassifier-inherited object
    """
    path = __file__[:-3]
    if model == "madry_lab_mnist_challenge":
        return dlm.models.CNNClassifier(
            activation=torch.nn.ReLU,
            attack=aml.pgd,
            attack_params=dict(alpha=0.01, epochs=40, epsilon=0.3),
            batch_size=50,
            classes=dlm.templates.mnist.classes,
            conv_kernel=5,
            conv_layers=(32, 64),
            conv_padding="same",
            dropout=0.0,
            epochs=100000,
            hidden_layers=(1024,),
            learning_rate=1e-4,
            loss=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer_params={},
            pool_kernel=2,
            scheduler=None,
            scheduler_params={},
            shape=dlm.templates.mnist.shape,
        ).load(f"{path}/madrylab_mnist_challenge.pt")
    raise AttributeError(f"module {__name__} has no attribute {model}")

"""
This module defines the state-of-the-art architectures and hyperparameters for
a variety of datasets from https://github.com/sheatsley/datasets. It serves as
a form of bookkeeping to be passed as arguments to model instantiations.
Author: Ryan Sheatsley and Blaine Hoak
Thu Feb 2 2023
"""
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class cicmalmem2022:
    """
    CIC-MalMem-2022 (https://www.unb.ca/cic/datasets/malmem-2022.html) is for
    predicting malware categories and benign applciations. It has four labels
    that describe malware categories (trojan horse, spyware, ransomware, or
    benign). The dataset was designed to test obfuscated malware detection
    methods through memory dumps. The state-of-the-art MLP accuracy is 61%
    (https://pdfs.semanticscholar.org/b2e2/0dc7a34753311472a5f2314fbf866d7eddd0.pdf).
    """

    classes = 4
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=128,
        classes=classes,
        dropout=0.0,
        epochs=5,
        hidden_layers=(32,),
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )


class fmnist:
    """
    Fashion-MNIST (https://github.com/zalandoresearch/fashion-mnist) is a
    dataset for predicting Zalando's article images. It has ten labels that
    describe particular articles of clothing, encoded as "0" through "9" (i.e.,
    t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag,
    ankle boot). The dataset was designed as a drop-in replacemnt for the
    original MNIST dataset for benchmarking machine learning algorithms. The
    state-of-the-art CNN accuracy is over 99%
    (https://arxiv.org/pdf/2001.00526.pdf).
    """

    classes = 10
    shape = (1, 28, 28)
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=64,
        classes=10,
        dropout=0.0,
        epochs=8,
        hidden_layers=(512, 256, 100),
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )
    cnn = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=64,
        classes=10,
        conv_layers=(16, 32),
        dropout=0.4,
        epochs=20,
        hidden_layers=(512,),
        kernel_size=3,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
        shape=shape,
    )


class nslkdd:
    """
    The NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html) contains extracted
    feature vectors from PCAPs that contain various information about traffic
    flows. It has five labels that describe benign traffic, denial-of-service
    attacks, network probes, user-to-root attacks, and remote-to-local attacks.
    The current state-of-the-art MLP accuracy is ~82%
    (https://www.ee.ryerson.ca/~bagheri/papers/cisda.pdf).
    """

    classes = 5
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=128,
        classes=classes,
        dropout=0.0,
        epochs=4,
        hidden_layers=(60, 32),
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )


class mnist:
    """
    MNIST (http://yann.lecun.com/exdb/mnist/) is a dataset for predicting
    handwritten digits. It has ten labels that describe a particular digit, "0"
    through "9". The state-of-the-art CNN accuracy is over 99%
    (https://arxiv.org/pdf/1710.09829.pdf).
    """

    classes = 10
    shape = (1, 28, 28)
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=64,
        classe=classes,
        dropout=0.0,
        hidden_layers=(512,),
        epochs=20,
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )
    cnn = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=64,
        classes=classes,
        conv_layers=(16, 32),
        dropout=0.4,
        epochs=20,
        hidden_layers=(128,),
        kernel_size=3,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
        shape=shape,
    )


class phishing:
    """
    The Phishing Dataset for Machine Learning: Feature Evaluation
    (https://www.fcsit.unimas.my/phishing-dataset) is a dataset for predicting
    whether a website is malicious based on features extracted from website DOMs
    and URLs. It has two classes, malicous and benign. The state-of-the-art MLP
    accuracy is 96%
    (https://www.sciencedirect.com/science/article/pii/S0020025519300763).
    """

    classes = 2
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=32,
        classes=classes,
        dropout=0.0,
        epochs=40,
        hidden_layers=(15,),
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )


class unswnb15:
    """
    The UNSW-NB15 (https://research.unsw.edu.au/projects/unsw-nb15-dataset) is
    a dataset for predicting network intrusions from a blend of real benign
    traffic with sythetically generated attacks. It contains ten classes, with
    nine different attacks families and benign. The state-of-the-art MLP
    accuracy is 81%
    (https://www.sciencedirect.com/science/article/pii/S0957417419300843).
    """

    classes = 10
    mlp = dict(
        activation=torch.nn.ReLU,
        attack=None,
        attack_params={},
        batch_size=128,
        classes=classes,
        dropout=0.0,
        epochs=40,
        hidden_layers=(15,),
        learning_rate=1e-2,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
    )

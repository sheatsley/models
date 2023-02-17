"""
This module runs unit tests to test performance and correctness of the dlm
repo. Specifically, this defines two types of tests: (1) functional, and (2)
performance. Details surrounding these tests can be found in the respecetive
classes: FunctionalTests and PerformanceTests.
Authors: Ryan Sheatsley and Blaine Hoak
Thu Feb 16 2023
"""

import aml  # ML robustness evaluations with PyTorch
import dlm  # PyTorch-based deep learning models with Keras-like interfaces
import itertools  # Functions creating iterators for efficient looping
import mlds  # Scripts for downloading, preprocessing, and numpy-ifying popular machine learning datasets
import unittest  # Unit testing framework
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration


class FunctionalTests(unittest.TestCase):
    """
    The following class implements functional tests. Functional correctness
    tests verify: (1) model classes are functionally correct (in that they do
    not cause errors), (2) special functionalities (determining the maximum
    batch size on gpus, performing adversarial training, using validation sets,
    loading and saving models, using checkpoints etc.) are correct.

    :func:`test_adversarial_training`: tests adversarial training subroutine
    :func:`test_checkpoints`: tests checkpoint subroutine correctness
    :func:`test_gpu_oom`: tests gpu oom prevention correctness
    :func:`test_save_load`: tests saving and loading models
    :func:`test_models`: test model class correctness
    :func:`verify_cnn`: verifies cnn architectures
    :func:`verify_linear`: verifies linear model architectures
    :func:`verify_mlp`: verifies mlp architectures
    """

    @classmethod
    def setUpClass(cls, classes=2, features=16, samples=32):
        """
        This function initializes the setup necessary for all test cases within
        this module. Specifically, this method: (1) generates data, (2)
        instantiates parameters used to configure models, and (3) sets an
        attack to be used for adversarial training.

        :param classes: number of classes to test with
        :type classes: int
        :param features: number of features to test with (should be power of 2)
        :type features: int
        :param samples: number of samples to test with
        :type samples: int
        :return: None
        :rtype: NoneType
        """
        cls.classes = classes
        cls.features = features
        cls.samples = samples
        cls.x = torch.rand((samples, features))
        cls.y = torch.randint(classes, (samples,))
        linear = dict(
            auto_batch=False,
            attack=None,
            attack_params=None,
            batch_size=samples // 2,
            classes=classes,
            epochs=3,
            learning_rate=1e-2,
            loss=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.SGD,
            optimizer_params=dict(momentum=0.01, nesterov=True),
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
            scheduler_params=dict(gamma=0.01),
            threads=-1,
            verbosity=0.25,
        )
        mlp = linear | dict(
            activation=torch.nn.ReLU,
            dropout=0.1,
            hidden_layers=(samples // classes, samples // classes),
        )
        cnn = mlp | dict(
            conv_layers=(features // 2, features),
            kernel_size=1,
            shape=(1, int(features ** (1 / 2)), int(features ** (1 / 2))),
        )
        cls.model_template_pairs = (
            (dlm.LinearClassifier, linear),
            (dlm.MLPClassifier, mlp),
            (dlm.CNNClassifier, cnn),
        )
        cls.attack = dict(
            attack=aml.bim, attack_params=dict(alpha=0.01, epsilon=0.1, epochs=2)
        )
        return None

    def test_adversarial_training(self):
        """
        This method validates the correctness of the adversarial training (and
        the use of a validation dataset). Specifically, a model is considered
        to be functionally correct if: (1) performing adversarial training, (2)
        creating validation data from the training set, and (3) using an
        explicitly passed validation set does not cause any errors.

        :return: None
        :rtype: NoneType
        """
        for model, template in self.model_template_pairs:
            print(f"Testing {model.__name__}...", end="\r")
            with self.subTest(Model=f"{model.__name__}"):
                model = model(**template | self.attack)
                model.fit(self.x, self.y, 0.10)
                model.fit(self.x, self.y, (self.x, self.y))
        return None

    def test_models(self):
        """
        This method validates the correctness of model instantiations and their
        operation. Specifically, a model object is considered to be
        functionally correct if: (1) it can be instantiated, (2) its
        architecture matches the arguments provided on initialization, and (3)
        its fit method does not cause any errors.

        :return: None
        :rtype: NoneType
        """
        for model, template in self.model_template_pairs:
            print(f"Testing {model.__name__}...", end="\r")
            with self.subTest(Model=f"{model.__name__}"):
                model = model(**template)
                model.fit(self.x, self.y)
                self.verify_linear(self.classes, self.features, model.model) if type(
                    model
                ) is dlm.LinearClassifier else self.verify_mlp(
                    self.classes,
                    self.features,
                    model.model,
                    template["activation"],
                    template["dropout"],
                    template["hidden_layers"],
                ) if type(
                    model
                ) is dlm.MLPClassifier else self.verify_cnn(
                    self.classes,
                    self.features,
                    model.model,
                    template["activation"],
                    template["conv_layers"],
                    template["dropout"],
                    template["hidden_layers"],
                    template["kernel_size"],
                    template["shape"],
                )
        return None

    def test_save_load(self, path="/tmp/"):
        """
        This method validates the correctness of model saving and loading
        subroutines. Specifically, these subroutines are considered to be
        functionally correct if models can be reinstantiated correctly after
        reading their structure from disk.

        :param path: path to save and load models
        :type path: str
        :return: None
        :rtype:NoneType
        """
        for (model_class, template), slim in itertools.product(
            self.model_template_pairs, (False, False)
        ):
            print(f"Testing {model_class.__name__}...", end="\r")
            with self.subTest(Model=f"{model_class.__name__}", Slim=slim):

                # initialize models at various states
                model = model_class(**template)
                trained = model_class(**template)
                skeleton = model_class(**template)
                model.fit(self.x, self.y)
                trained.fit(self.x, self.y)

                # restore from full model and from state dicts
                savepath = f"{path}dlm_test_{model_class.__name__}"
                model.save(savepath, slim=slim)
                trained.load(savepath)
                skeleton.load(savepath)
                self.assertMultiLineEqual(
                    model.model.__repr__() * 2,
                    trained.model.__repr__() + skeleton.model.__repr__(),
                )
        return None

    def verify_cnn(
        self,
        classes,
        features,
        model,
        activation,
        conv_layers,
        dropout,
        hidden_layers,
        kernel_size,
        shape,
    ):
        """
        This method verfies convolutional neural networks. Specifically, it
        confirms that: (1) each layer follows a
        dropout-convolution-activation-maxpool pattern a
        convolutional-layer-number of times, followed by a flatten into a
        dropout-linear-activation pattern a hidden-layer-number of times, (2)
        dropout rates set to those passed on object instantiation, and (3)
        convolutional input & output channels and linear layer input & output
        features match input shape, convolutional layers, hidden layers, and
        classes.

        :param classes: number of classes
        :type classes: int
        :param features: number of features
        :type features: int
        :param model: model to verify
        :type model: torch.nn Sequential object
        :param activation: activation function used
        :type activation: torch.nn module class
        :param conv_layers: number of filters at each convolutional layers
        :type conv_layers: tuple of ints
        :param dropout: dropout rate
        :type dropout: float
        :param hidden_layers: number of neurons at each hidden layers
        :type hidden_layers: tuple of ints
        :param kernel_size: size of the convolving kernel
        :type kernel_size: int
        :param shape: original input shape (channels, width, height)
        :type shape: tuple of ints
        :return: None
        :rtype: NoneType
        """

        # validate cnn portion of the network
        cnn_comp = itertools.cycle(
            (torch.nn.Dropout, torch.nn.Conv2d, activation, torch.nn.MaxPool2d)
        )
        cnn_params = itertools.chain(
            *itertools.product((dropout,), conv_layers, (None,), (kernel_size,))
        )
        for layer, comp, param in zip(
            model[1 : len(conv_layers) * 4 + 1], cnn_comp, cnn_params
        ):
            if comp is torch.nn.Dropout:
                self.assertTupleEqual((type(layer), layer.p), (comp, param))
            elif comp is torch.nn.Conv2d:
                self.assertTupleEqual((type(layer), layer.out_channels), (comp, param))
            elif comp is torch.nn.MaxPool2d:
                self.assertTupleEqual((type(layer), layer.kernel_size), (comp, param))
            else:
                self.assertEqual(type(layer), comp)

        # validate mlp portion of the network
        mlp_comp = itertools.cycle((torch.nn.Dropout, torch.nn.Linear, activation))
        mlp_params = itertools.chain(
            *itertools.product((dropout,), hidden_layers, (None,))
        )
        for layer, comp, param in zip(
            model[len(conv_layers) * 4 + 2 : -1], mlp_comp, mlp_params
        ):
            if comp is torch.nn.Dropout:
                self.assertTupleEqual((type(layer), layer.p), (comp, param))
            elif comp is torch.nn.Linear:
                self.assertTupleEqual((type(layer), layer.out_features), (comp, param))
            else:
                self.assertEqual(type(layer), comp)

        # validate input and output features
        self.assertTupleEqual(
            (model[0].unflattened_size, model[-1].out_features), (shape, classes)
        )
        return None

    def verify_linear(self, classes, features, model):
        """
        This method verfies linear models. Specifically, it confirms
        that the number of inputs and outputs matches the number of
        features and classes, respectively.

        :param classes: number of classes
        :type classes: int
        :param features: number of features
        :type features: int
        :param model: model to verify
        :type model: torch.nn Sequential object
        :return: None
        :rtype: NoneType
        """
        self.assertTupleEqual(
            (model[0].in_features, model[0].out_features), (features, classes)
        )
        return None

    def verify_mlp(self, classes, features, model, activation, dropout, hidden_layers):
        """
        This method verfies multi-layer perceptrons. Specifically, it confirms
        that: (1) each layer follows a dropout-linear-activation pattern, (2)
        dropout rates set to those passed on object instantiation, and (3)
        linear layer input and output features match input features, hidden
        layers, and classes.

        :param classes: number of classes
        :type classes: int
        :param features: number of features
        :type features: int
        :param model: model to verify
        :type model: torch.nn Sequential object
        :param activation: activation function used
        :type activation: torch.nn module class
        :param dropout: dropout rate
        :type dropout: float
        :param hidden_layers: number of neurons at each hidden layers
        :type hidden_layers: tuple of ints
        :return: None
        :rtype: NoneType
        """
        comps = itertools.cycle((torch.nn.Dropout, torch.nn.Linear, activation))
        params = itertools.chain(*itertools.product((dropout,), hidden_layers, (None,)))
        for layer, comp, param in zip(model[:-1], comps, params):
            if comp is torch.nn.Dropout:
                self.assertTupleEqual((type(layer), layer.p), (comp, param))
            elif comp is torch.nn.Linear:
                self.assertTupleEqual((type(layer), layer.out_features), (comp, param))
            else:
                self.assertEqual(type(layer), comp)
        self.assertTupleEqual(
            (model[1].in_features, model[-1].out_features), (features, classes)
        )
        return None

"""
This module defines functions to instantiate PyTorch-based deep learning models
with scikit-learn-like interfaces.
Author: Ryan Sheatsley & Blaine Hoak
Mon Oct 24 2022
"""
import vgg

import attacks  # PyTorch-based framework for attacking deep learning models
import itertools  # Functions creating iterators for efficient looping
import loss as libloss  # PyTorch-based custom loss functions
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
from utilities import print  # Use timestamped print
import optimizers

# TODO
# add vgg
# add resnet
# consider better debugging when field in architectures is not specified (eg adv_train)
# remove attack object instantiation when Attacks module is updated
# reconsider inheritence structure (all args must be passed to each class atm)
# investigate if https://github.com/pytorch/pytorch/issues/8741 works for cpu method
# __repr__
# implement auto-batching for GPU memory


class LinearClassifier:
    """
    This class defines a PyTorch-based linear model. It exposes the basic
    interfaces useful for building broadly any neural network. While primarily
    designed to be inherited for more complex models, instantiating this class
    with defualt parameters yields a single-layer linear model with categorical
    cross-entropy loss (i.e., softmax regression). It defines the following
    methods:

    :func:`__init__`: instantiates LinearClassifier objects
    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch nn.Sequential object attributes
    :func:`__repr__`: returns architecture, hyperparameters, and algorithms
    :func:`accuracy`: returns model accuracy
    :func:`build`: assembles the model
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training
    :func:`load`: loads model, optimizer, and scheduler
    :func:`predict`: returns predicted labels
    :func:`save`: saves model, optimizer, and scheduler
    """

    def __init__(
        self,
        batch_size=128,
        iters=10,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=optimizers.Adam,
        optimizer_params={},
        scheduler=None,
        scheduler_params={},
        threads=None,
        verbosity=0.25,
    ):
        """
        This method instantiates LinearClassifier objects with a variety of
        attributes necessary to support model training and logging.
        Importantly, models are not usable until they are trained (via fit()),
        this "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to scikit-learn
        model classes) on initialization.

        :param batch_size: training batch size (-1 for 1 batch)
        :type batch_size: int
        :param iters: number of training iterations (ie epochs)
        :type iters: int
        :param learning_rate: learning rate
        :type learning_rate: float
        :param loss: loss function
        :type loss: torch nn class
        :param optimizer: optimizer from optimizers package
        :type optimizer: torch optim class
        :param optimizer_params: optimizer parameters
        :type optimizer_params: dict
        :param scheduler: scheduler for dynamic learning rates
        :type scheduler: torch optim.lr_scheduler class or None
        :param scheduler_params: scheduler parameters
        :type scheduler_params: dict
        :param threads: number of cpu threads used for training
        :type threads: int or None (for max threads)
        :param verbosity: print the loss every verbosity%
        :type verbosity: float
        :return: a linear classifier skeleton
        :rtype: LinearClassifier object
        """
        super().__init__()
        self.batch_size = batch_size
        self.iters = iters
        self.learning_rate = learning_rate
        self.loss_func = loss(reduction="sum")
        self.optimizer_alg = optimizer
        self.scheduler_alg = scheduler
        self.threads = (
            min(threads, torch.get_num_threads())
            if threads
            else torch.get_num_threads()
        )
        self.verbosity = max(1, int(iters * verbosity))
        self.params = {
            "batch_size": batch_size,
            "iters": iters,
            "lr": learning_rate,
            "loss": loss.__name__,
            "optim": optimizer.__name__,
            "state": "skeleton",
        } | ({"lr_scheduler": scheduler.__name__} if scheduler else {})
        return None

    def __call__(self, x, grad=True):
        """
        This method returns the model logits. Optionally, gradient-tracking can
        be disabled for fast inference. Importantly, when using gpus, inputs
        are auto-batched to sizes that will fit within VRAM to prevent
        out-of-memory errors.

        :param x: the batch of inputs
        :type x: torch Tensor object (n, m)
        :param grad: whether gradient computation
        :type grad: boolean
        :return: model logits
        :rtype: torch Tensor object (n, c)
        """
        with torch.set_grad_enabled(grad):
            return self.model(x)

    def __getattr__(self, name):
        """
        This method ostensibly aliases torch nn.Sequential object (i.e.,
        self.model) attributes to be accessible by this object directly. It is
        principally used for easier debugging.

        :param name: name of the attribute to recieve from self.model
        :type name: str
        :return: the desired attribute (if it exists)
        :rtype: misc
        """
        return self.model.__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of algorithms,
        hyperparameters, and architecture.

        :return: algorithms used and hyperparameters
        :rtype: str
        """
        return f"LinearClassifer({self.params})"

    def accuracy(self, x, y):
        """
        This method returns the fraction of inputs classified correctly over
        the total number of samples. Additionally, a boolean tensor containing
        which inputs were classified correctly is stored (which is useful for
        attacks that leverage this information, e.g., APGD, as shown in
        https://arxiv.org/abs/2003.01690).

        :param x: batch of inputs
        :type x: torch Tensor object (n, m)
        :param y: batch of labels
        :type y: Pytorch Tensor object (n,)
        :return: model accuracy
        :rtype: torch Tensor object (1,)
        """
        correct = self(x, False).argmax(1).eq(y)
        self.correct = correct
        return correct.sum().div(y.numel())

    def build(self, x, y):
        """
        This method instantiates a torch Sequential object. This abstraction
        allows us to dynamically build models based on the passed-in dataset,
        versus hardcoding model architectures via a forward() method. Moreover,
        we augment the stored parameters for informative string
        representations.

        :param x: dataset of inputs
        :type x: torch Tensor object (n, m)
        :param y: dataset of labels
        :type y: Pytorch Tensor object (n,)
        :return: an untrained linear classifier
        :rtype: torch Sequential object
        """
        self.params["features"] = x.size(1)
        self.params["classes"] = y.unique().numel()
        self.params["state"] = "untrained"
        return torch.nn.Sequential(torch.nn.Linear(x.size(1), y.unique().numel()))

    def cpu(self):
        """
        This method moves the model, optimizer, and scheduler to the cpu. At
        this time, to() and cpu() methods are not supported for optimizers nor
        schedulers (https://github.com/pytorch/pytorch/issues/41839), so we
        leverage a trick shown in
        https://github.com/pytorch/pytorch/issues/8741 to refresh the state of
        optimizers and schedulers (as these subroutines match device state to
        that of the attached parameters).

        :return: linear classifier
        :rtype: LinearClassifier object
        """
        self.model.cpu()
        self.cuda_or_cpu()
        return self

    def cuda(self):
        """
        This method moves the model, optimizer, and scheduler to the gpu. At
        this time, to() and cuda() methods are not supported for optimizers nor
        schedulers (https://github.com/pytorch/pytorch/issues/41839), so we
        leverage a trick shown in
        https://github.com/pytorch/pytorch/issues/8741 to refresh the state of
        optimizers and schedulers (as these subroutines match device state to
        that of the attached parameters).

        :return: linear classifier
        :rtype: LinearClassifier object
        """
        self.model.cuda()
        self.cuda_or_cpu()
        return self

    def cuda_or_cpu(self):
        """
        This method applies the trick shown in
        https://github.com/pytorch/pytorch/issues/8741 to optimizer and
        scheduler states. Given that process is device agnostic (in that the
        states are simply refreshed), it is expected that method be called by
        either cpu() or cuda().

        :return: None
        :rtype: NoneType
        """
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        self.scheduler.load_state_dict(self.scheduler.state_dict())
        return None

    def fit(self, x, y, validation=0.0, atk=None):
        """
        This method is the heart of all LinearClassifier-inherited objects. It
        performs three functions: (1) instantiating a model (i.e., torch
        Sequential object) and updates the string representation based on
        attributes of the passed in dataset (i.e., x and y), (2) training the
        instantiated model, (3) optionally performing adversarial training, and
        (4) computing statistics over a validation set.

        :param x: training inputs
        :type x: torch Tensor object (n, m)
        :param y: training labels
        :type y: Pytorch Tensor object (n,)
        :param validation: hold-out set or proportion of training data use
        :type validation: tuple of torch Tensor objects or float
        :param atk: attack to use to perform adversarial training
        :type atk: clevertorch attack object
        :return: a trained linear classifier model
        :rtype: LinearClassifier object
        """

        # instantiate model, optimizer, and scheduler
        self.model = self.build(x, y)
        self.optimizer = self.optimizer_alg(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.scheduler = (
            self.scheduler_alg(self.optimizer, **self.scheduler_params)
            if self.scheduler_alg
            else None
        )
        print(f"Defined model:\n{self.model}")

        # prepare validation set (if applicable) and data loader
        if isinstance(validation, float):
            num_val = int(y.numel() * validation)
            perm_idx = torch.randperm(y.numel())
            x_val, x = x[perm_idx].tensor_split([num_val])
            y_val, y = y[perm_idx].tensor_split([num_val])
            print(f"Validation set created with shape: {x_val.size()} × {y.size()}")
        else:
            x_val, y_val = validation
        trainset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        print(f"Data loader prepared with shape: {x.size()} × {y.size()}")

        # attach attack if performing adversarial training
        if atk:
            self.atk = atk
            self.atk.model = self
            self.params["atk"] = repr(self.atk)
            print(f"Performing adversarial training with: {self.atk}")

        # set cpu threads, metadata variables, and track gradients
        device = (
            torch.cuda.get_device_name()
            if self.model.is_cuda
            else f"cpu ({self.threads} threads)"
        )
        print(f"Training for {self.iters} iterations on {device}...")
        self.model.train()
        self.model.requires_grad_(True)
        self.stats = {"train_acc": [], "train_loss": []} | (
            {"val_acc": [], "val_loss": []} if validation != 0 else {}
        )
        max_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)

        # enter main training loop; apply scheduler and reset iteration loss
        for current_iter in range(self.iters):
            self.scheduler.step() if self.scheduler else None
            iter_loss = 0
            iter_acc = 0

            # perform one iteration of training
            for xb, yb in trainset:
                self.model.requires_grad_(False)
                xb = self.atk.craft(xb, yb) if atk else xb
                self.model.requires_grad_(True)
                batch_logits = self.model(xb)
                batch_loss = self.loss(batch_logits, yb)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                iter_loss += batch_loss.item()
                iter_acc += batch_logits.argmax(1).eq(yb).sum()

            # collect learning statistics every iteration
            self.stats["train_loss"].append(iter_loss)
            self.stats["train_acc"].append(iter_acc.div(y.numel()).item())
            if validation != 0:
                val_logits = self.model(x_val, False)
                self.stats["val_loss"].append(self.loss(val_logits, y_val))
                self.stats["val_acc"].append(
                    val_logits.argmax(1).eq(y_val).div(y_val.numel())
                )

            # print learning statistics every verbosity%
            print(
                f"Iter: {current_iter + 1}/{self.iters}",
                f"Loss: {iter_loss:.3f} ({iter_loss - self.stats['train_loss']:+.3})",
                f"Accuracy: {iter_acc:.3f} ({iter_acc - self.stats['train_acc']:+.3})",
                (f"Validation Loss: {")
            ) if not current_iter % self.info else None

        # disable gradients, restore thread count, and compute adversarial statistics
        print(
            "Disabling gradients",
            ("and setting max threads to", f"{max_threads}...")
            if not self.model.is_cuda
            else "...",
        )
        self.model.requires_grad_(False)
        torch.set_num_threads(max_threads)
        self.model.eval()
        if atk:
            atk_logits = self.model(self.atk.craft(x, y))
            atk_loss = self.loss(atk_logits, y).item()
            adv_acc = atk_logits.argmax(1).eq(yb).sum().div(y.numel()).item()
            print("Adversarial Loss")

        # print statistics about adversarial training (if not CIFAR 10)
        if adv_train and x.size(-1) != 3072:
            train_adv = self.adv_train_atk.craft(x, y)
            adv_loss = self.loss(self.model(train_adv), y)
            adv_acc = self.accuracy(train_adv, y).item()
            print(f"Adversarial loss: {adv_loss:.2f} & accuracy: {adv_acc:.2f}")
        print(f"Freezing parameters and restoring thread count to {max_threads}...")
        return self


class LinearClassifier(torch.nn.Module):
    """
    This class defines a simple PyTorch-based linear model class that contains
    the fundamental components of any deep neural network. While this is class
    is designed to be inherited to provide simple sub-classes for multi-layer
    linear or convolutional models, instantiating this class with default
    parameters instantiates a single-layer linear model with categorical
    cross-entropy loss. It defines the following methods:

    :func:`__init__`: initial object setup
    :func:`__call__`: returns model logits
    :func:`accuracy`: computes the accuracy of the model on data
    :func:`cpu`: moves all tensors to the cpu
    :func:`build`: assembles the PyTorch model
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels
    """

    def fit(self, x, y, xtest_validation, adv_train=False):
        """
        This method prepends and appends linear layers with dimensions
        inferred from the dataset. Moreover, it trains the model using the
        paramterized optimizer.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :param dataset: samples with associated labels
        :type dataset: (n x m, n) tuple of tensors as a TensorDataset
        :return: a trained model
        :rtype: LinearClassifier object
        """

        # instantiate learning model and save dimensionality
        self.model = self.build(x, y)
        self.features = x.size(1)
        print(f"Defined model:\n{self.model}")

        # configure optimizer and load data into data loader
        print(f"Preparing data loader of shape: {x.size()} x {y.size()}")

        # hacky fix for cifar 10
        if x.size(-1) == 3072:
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.epochs
            )
        else:
            self.opt = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda" if x.is_cuda else "cpu"),
        )

        # attatch loss and model parameters to attacks for adversarial training
        if adv_train:
            self.adv_train_atk = attacks.Attack(
                **{
                    **self.adv_train,
                    **{
                        "model": self,
                        "loss": libloss.Loss(
                            type(self.loss), max_obj=True, x_req=False, reduction="none"
                        ),
                    },
                }
            )
            print(
                "Performing",
                self.adv_train_atk.name,
                "adversarial training!",
                f"({self.adv_train_atk.traveler.epochs} epochs,",
                f"alpha={self.adv_train_atk.traveler.alpha})",
            )

        # main training loop
        print(f"Unfreezing parameters and limiting thread count to {self.threads}...")
        self.model.train()
        self.freeze(False)
        max_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        for epoch in range(self.epochs):
            eloss = 0
            for sample_batch, label_batch in dataset:

                # disable grad tracking for model parameters
                if adv_train:
                    self.freeze(True)
                    sample_batch = self.adv_train_atk.craft(sample_batch, label_batch)
                    self.adv_train_atk.traveler.init_req = True
                    self.freeze(False)
                loss = self.loss(
                    self.model(sample_batch),
                    label_batch,
                )
                loss.backward()
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                eloss += loss.item()

            # show model loss every info-percent
            if not epoch % self.info:
                print(f"Loss at epoch {epoch}: {eloss}")
            self.scheduler.step() if hasattr(self, "scheduler") else None
        accuracy = self.accuracy(x, y).item()
        print(f"Final loss: {eloss:.2f}, & accuracy: {accuracy:.2f}")

        # print statistics about adversarial training (if not CIFAR 10)
        if adv_train and x.size(-1) != 3072:
            train_adv = self.adv_train_atk.craft(x, y)
            adv_loss = self.loss(self.model(train_adv), y)
            adv_acc = self.accuracy(train_adv, y).item()
            print(f"Adversarial loss: {adv_loss:.2f} & accuracy: {adv_acc:.2f}")
        print(f"Freezing parameters and restoring thread count to {max_threads}...")
        torch.set_num_threads(max_threads)
        self.freeze(True)
        self.model.eval()
        return self

    def freeze(self, freeze=True):
        """
        This method iterates over all model parameters and disables tracking
        operation history within the autograd system. This is particularly
        useful in cases with adversarial machine learning: operations on inputs
        are tracked (as opposed to model parameters when training) and
        gradients with respect to these inputs are computed. Since they are
        multiple backward()'s calls in these scenarios, we can achieve
        (sometimes significant) performance gains by explicitly disabling
        autograd history for model parameters.

        :param freeze: enable or disable autograd history for model parameters
        :type freeze: boolean
        :return: the model itself
        :rtype: LinearClassifier object
        """
        for param in self.model.parameters():
            param.requires_grad = not freeze
        return self

    def load(self, path, x, y):
        """
        This method loads a set of model parameters, saved via torch.save
        method. Since this class defines PyTorch models via the build method,
        we also require the training data and labels to properly infer input
        feautres and output labels.

        :param path: path to the saved model parameters
        :type path: string
        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: the model itself
        :rtype: LinearClassifier object
        """
        self.model = self.build(x, y)
        self.features = x.size(1)
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model:\n{self.model}")
        return self

    def predict(self, x):
        """
        This method serves as a wrapper for __call__ to behave like predict
        methods in scikit-learn model classes by returning an class label
        (instead of model logits).

        :param x: samples
        :type x: n x m tensor of samples
        :return: predicted labels
        :rtype: n-length tensor of integers
        """
        return torch.argmax(self(x, grad=False), dim=1) if len(x) else torch.tensor([])


class MLPClassifier(LinearClassifier):
    """
    This class inherits from LinearClassifier and instantiates a PyTorch-based
    multi-layer (if the number of hidden layers is non-zero, else a
    single-layer is returned) perceptron classifier. Specifically, it inherits
    the following methods as-is from LinearClassifier:

    :func:`__init__`: initial class setup
    :func:`__call__`: returns model logits
    :func:`cpu`: moves all tensors to the cpu
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels

    It redefines the following methods:

    :func:`build`: assembles the PyTorch MLP model
    """

    def __init__(
        self,
        hidden_layers=(15,),
        activation=torch.nn.ReLU,
        adv_train=None,
        optimizer=optimizers.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This function describes the initial setup for a multi-layer perceptron
        classifier. Importantly, models are not usable until they are trained
        (via fit()), this "lazy" model creation schema allows us to abstract
        out parameterization of number of attributes and labels (thematically
        similar to scikit-learn model classes).

        :param hidden_layers: the number of neurons at each layer
        :type hidden_layers: i-length tuple of integers
        :param activation: activation functions from torch.nn.functional
        :type activation: callable
        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: supports optimizers in optimizers package
        :type optimizer: callable from optimizers
        :param loss: supports loss functions in torch.nn
        :type loss: callable from torch.nn.modules.loss
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss at % itervals
        :type info: float between 0 and 1
        :return: mutli-layer linear classifier
        :rtype: MLPClassifier object
        """
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        self.hidden_layers = hidden_layers
        self.activation = activation
        return None

    def build(self, x, y):
        """
        This method overrides the implementation of build() from the
        LinearClassifier class. Specifically, it adds support for hidden
        layers and activation functions when instantiating a PyTorch sequential
        container. The provided abstraction allows us to dynamically build
        models based on the passed-in dataset, as opposed to hardcoding model
        architectures via defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: multi-layer perceptron model
        :rtype: Sequential container
        """

        # compute the number of classes (needed now if no hidden layers)
        labels = torch.unique(y).size(0)

        # instantiate initial & hidden linear layers
        neurons = (x.size(1),) + self.hidden_layers
        linear = (
            [
                torch.nn.Linear(neurons[i], neurons[i + 1])
                for i in range(len(neurons) - 1)
            ]
            if len(self.hidden_layers)
            else [torch.nn.Linear(x.size(1), labels)]
        )

        # interleave linear layers with activation function
        layers = itertools.product(
            linear,
            [self.activation()],
        )

        # add output layer and instantiate sequential container
        return torch.nn.Sequential(
            *itertools.chain(
                itertools.chain.from_iterable(layers),
                [torch.nn.Linear(self.hidden_layers[-1], labels)]
                if len(self.hidden_layers)
                else [],
            )
        )


class CNNClassifier(LinearClassifier):
    """
    This class inherits from LinearClassifier and instantiates a PyTorch-based
    convolutional neural network classifier. Specifically, it inherits the
    following methods as-is from LinearClassifier:

    :func:`__init__`: initial class setup
    :func:`__call__`: returns model logits
    :func:`cpu`: moves all tensors to the cpu
    :func:`fit`: trains the model
    :func:`freeze`: disable autograd tracking of model parameters
    :func:`load`: loads saved model parameters
    :func:`predict`: returns predicted labels

    It redefines the following methods:

    :func:`build`: assembles the PyTorch CNN model
    """

    def __init__(
        self,
        conv_layers,
        shape=None,
        stride=1,
        kernel_size=3,
        linear_layers=(15,),
        drop_prob=0.5,
        activation=torch.nn.ReLU,
        adv_train=None,
        optimizer=optimizers.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This function describes the initial setup for a  convolutional neural
        network classifier. Importantly, models are not usable until they are
        trained (via fit()), this "lazy" model creation schema allows us to
        abstract out parameterization of number of attributes and labels
        (thematically similar to scikit-learn model classes).

        :param conv_layers: the number of filters at the ith layer
        :type conv_layers: i-length tuple
        :param shape: the expected shape of the input image
        :type shape: tuple of form: (channels, width, height)
        :param stride: stride length for the convolutional layers
        :type stride: integer
        :param kernel_size: kernel size for all convolutional layers
        :type kernel_size: integer
        :param linear_layers: the number of neurons at each linear layer
        :type linear_layers: i-length tuple of integers
        :param drop_prob: probability of an element to be zeroed (omitted if 0)
        :type drop_prob: float
        :param activation: activation functions from torch.nn.functional
        :type activation: callable
        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: supports optimizers in optimizers package
        :type optimizer: callable from optimizers
        :param loss: supports loss functions in torch.nn
        :type loss: callable from torch.nn.modules.loss
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss at % itervals
        :type info: float between 0 and 1
        :return: convolutional neural network classifier
        :rtype: CNNClassifier object
        """
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.linear_layers = linear_layers
        self.activation = activation
        self.shape = shape

        # set popular maxpool params & enable benchmarks
        self.mp_ksize = 2
        self.mp_stride = 2
        torch.backends.cudnn.benchmark = True
        return None

    def build(self, x, y):
        """
        This method overrides the implementation of build() from the
        LinearClassifier class. Specifically, it adds support for
        convolutional, dropout, and hidden layers, as well as activation
        functions when instantiating a PyTorch sequential container. The
        provided abstraction allows us to dynamically build models based on the
        passed-in dataset, as opposed to hardcoding model architectures via
        defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: convolutional model
        :rtype: Sequential container
        """

        # compute the number of classes and output of last maxpool
        labels = torch.unique(y).size(0)
        if not self.shape:
            self.shape = x.size()[1:]
            x = x.flatten()
        last_maxout = (
            self.shape[1] // self.mp_ksize ** len(self.conv_layers)
        ) ** 2 * self.conv_layers[-1]

        # instantiate convolutional layers
        conv_layers = (self.shape[0],) + self.conv_layers
        conv_layers = [
            torch.nn.Conv2d(
                conv_layers[i],
                conv_layers[i + 1],
                self.kernel_size,
                self.stride,
                "same",
            )
            for i in range(len(conv_layers) - 1)
        ]

        # instantiate linear layers
        neurons = (last_maxout,) + self.linear_layers
        linear_layers = (
            [
                torch.nn.Linear(neurons[i], neurons[i + 1])
                for i in range(len(neurons) - 1)
            ]
            if len(self.linear_layers)
            else [torch.nn.Linear(last_maxout, labels)]
        )

        # interleave initial and convolution layers with activation and maxpool
        cnn_layers = itertools.product(
            conv_layers,
            [self.activation()],
            [
                torch.nn.MaxPool2d(kernel_size=self.mp_ksize, stride=self.mp_stride),
            ],
        )

        # interleave output of maxpool and linear layers with activation
        mlp_layers = itertools.product(
            linear_layers,
            [self.activation()],
        )

        # concatenate cnn & mlp layers, add dropout, and return the container
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, self.shape),
            *itertools.chain.from_iterable(cnn_layers),
            torch.nn.Flatten(),
            torch.nn.Dropout(self.drop_prob, inplace=True),
            *itertools.chain(
                itertools.chain.from_iterable(mlp_layers),
                [torch.nn.Linear(self.linear_layers[-1], labels)]
                if len(self.linear_layers)
                else [],
            ),
        )


class vggwrapper(LinearClassifier):
    def __init__(
        self,
        adv_train={
            "epochs": 3,
            "optimizer": optimizers.SGD,
            "alpha": 0.01,
            "random_alpha": 0.03,
            "change_of_variables": False,
            "saliency_map": "identity",
            "norm": float("inf"),
            "jacobian": "model",
        },
        optimizer=optimizers.SGD,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=5e-2,
        iters=300,
        threads=None,
        info=0,
    ):

        print("Using VGG wrapper:", learning_rate, batch_size, iters)
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        return None

    def build(self, x, y):
        return vgg.vgg16()


class VGG16Classifier(LinearClassifier):
    """this is apparently broken (does not converge) will investigate later"""

    def __init__(
        self,
        conv_layers=(3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512),
        shape=None,
        stride=1,
        kernel_size=3,
        padding=1,
        linear_layers=(512, 512),
        drop_prob=0.5,
        activation=torch.nn.ReLU,
        adv_train=None,
        optimizer=optimizers.Adam,
        loss=torch.nn.CrossEntropyLoss,
        batch_size=128,
        learning_rate=1e-4,
        iters=10,
        threads=None,
        info=0.25,
    ):
        """
        This function describes the initial setup for a VGG16 classifier.
        Importantly, models are not usable until they are trained (via fit()),
        this "lazy" model creation schema allows us to abstract out
        parameterization of number of attributes and labels (thematically
        similar to scikit-learn model classes).

        :param conv_layers: the number of filters at the ith layer
        :type conv_layers: i-length tuple
        :param shape: the expected shape of the input image
        :type shape: tuple of form: (channels, width, height)
        :param stride: stride length for the convolutional layers
        :type stride: integer
        :param kernel_size: kernel size for all convolutional layers
        :type kernel_size: integer
        :param padding: controls the amount of padding applied to the input
        :type padding: int
        :param linear_layers: the number of neurons at each linear layer
        :type linear_layers: i-length tuple of integers
        :param drop_prob: probability of an element to be zeroed (omitted if 0)
        :type drop_prob: float
        :param activation: activation functions from torch.nn.functional
        :type activation: callable
        :param adv_train: attack used for adversarial training
        :type adv_train: Attack object
        :param optimizer: supports optimizers in optimizers package
        :type optimizer: callable from optimizers
        :param loss: supports loss functions in torch.nn
        :type loss: callable from torch.nn.modules.loss
        :param batch_size: size of minibatches
        :type batch_size: integer
        :param learning_rate: learning rate schedule
        :type learning_rate: float
        :param iters: number of training iterations (ie epochs)
        :type iters: integer
        :param threads: sets the threads used for training
        :type threads: integer
        :param info: print the loss at % itervals
        :type info: float between 0 and 1
        :return: convolutional neural network classifier
        :rtype: VGG16Classifier object
        """
        super().__init__(
            adv_train, optimizer, loss, batch_size, learning_rate, iters, threads, info
        )
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_prob = drop_prob
        self.linear_layers = linear_layers
        self.activation = activation
        self.shape = shape

        # set popular maxpool params & enable benchmarks
        self.mp_ksize = 2
        self.mp_stride = 2
        torch.backends.cudnn.benchmark = True
        return None

    def build(self, x, y):
        """
        This method overrides the implementation of build() from the
        LinearClassifier class. Specifically, it adds support for
        convolutional, dropout, and hidden layers, as well as activation
        functions when instantiating a PyTorch sequential container. The
        provided abstraction allows us to dynamically build models based on the
        passed-in dataset, as opposed to hardcoding model architectures via
        defining a "forward" method.

        :param x: dataset of samples
        :type x: n x m matrix
        :param y: dataset of labels
        :type y: n-length vector
        :return: convolutional model
        :rtype: Sequential container
        """

        # compute the number of classes
        labels = torch.unique(y).size(0)
        if not self.shape:
            self.shape = x.size()[1:]
            x = x.flatten()

        # instantiate convolutional layers
        conv_layers = self.conv_layers
        conv_layers = [
            torch.nn.Conv2d(
                conv_layers[i],
                conv_layers[min(i + 1, len(conv_layers) - 1)],
                self.kernel_size,
                self.stride,
                self.padding,
            )
            for i in range(len(conv_layers))
        ]

        # instantiate linear layers
        neurons = self.linear_layers
        linear_layers = [
            torch.nn.Linear(neurons[i], neurons[i + 1]) for i in range(len(neurons) - 1)
        ]

        # interleave initial and convolution layers with activation
        cnn_layers = list(
            itertools.product(conv_layers, [self.activation(inplace=True)])
        )

        # maxpool after 2nd, 4th, 7th, 10th, and 13th convolution
        max_inserts = [2, 5, 9, 13, 17]
        for insert in max_inserts:
            cnn_layers = (
                cnn_layers[:insert]
                + [
                    (
                        torch.nn.MaxPool2d(
                            kernel_size=self.mp_ksize, stride=self.mp_stride
                        ),
                    )
                ]
                + cnn_layers[insert:]
            )

        # interleave linear layers with activation and dropout
        """
        mlp_layers = list(
            itertools.product(
                linear_layers,
                [self.activation(True)],
                [torch.nn.Dropout(self.drop_prob)],
            )
        )
        """

        # concatenate cnn & mlp layers and return the container
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, self.shape),
            *itertools.chain.from_iterable(cnn_layers),
            torch.nn.Flatten(),
            # *itertools.chain.from_iterable(mlp_layers),
            # torch.nn.Linear(neurons[-1], labels),
            torch.nn.Linear(512, labels),
        )


if __name__ == "__main__":
    """
    Example usage with training MNIST MLP and CNN models.
    """
    import architectures  # optimal PyTorch-based model architectures and hyperparameters
    import sklearn.metrics  # Score functions, performance metrics, and pairwise metrics and distance computations
    import utilities  # Various utility functions

    # load dataset
    use_cuda, device = (True, "cuda:0") if torch.cuda.is_available() else (False, "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor) if use_cuda else None
    tx, ty, vx, vy = utilities.load("mnist", device=device)

    # train linear model
    print("Training MNIST Linear Classifier...")
    linear_model = LinearClassifier(
        iters=6,
        learning_rate=1e-3,
    ).fit(tx, ty)
    print(
        "Linear Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), linear_model.predict(tx).cpu()),
        "\nLinear Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), linear_model.predict(vx).cpu()),
    )

    # train mlp model
    print("Training MNIST MLP Classifier...")
    mlp_model = MLPClassifier(**architectures._mnist(cnn=False)).fit(tx, ty)
    print(
        "MLP Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), mlp_model.predict(tx).cpu()),
        "\nMLP Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), mlp_model.predict(vx).cpu()),
    )

    # train cnn model
    print("Training MNIST CNN Classifier...")
    cnn_model = CNNClassifier(**architectures.mnist).fit(tx, ty)
    print(
        "CNN Training Report:\n",
        sklearn.metrics.classification_report(ty.cpu(), cnn_model.predict(tx).cpu()),
        "\nCNN Testing Report:\n",
        sklearn.metrics.classification_report(vy.cpu(), cnn_model.predict(vx).cpu()),
    )
    raise SystemExit(0)

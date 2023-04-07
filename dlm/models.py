"""
This module defines the core for the deep learning models repo. All supported
deep learning models are defined here, with the LinearClassifier class defining
the primary features across all models (which all inherit from
LinearClassifier). Much like Keras, these classes define simple interfaces to
support rapid prototyping of standard deep learning models.
Author: Ryan Sheatsley & Blaine Hoak
Thu Feb 2 2023
"""
import itertools

import pandas
import torch

# TODO
# update all hyperparameters
# add trades loss
# add mart loss
# add gtsrb hparams
# add cifar10 hparams


class LinearClassifier:
    """
    This class defines a PyTorch-based linear model. It exposes the basic
    interfaces useful for building broadly any neural network. While primarily
    designed to be inherited for more complex models, this class can be
    instantiated to build, for example, single-layer softmax regression models.

    :func:`__init__`: instantiates LinearClassifier objects
    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch.nn Sequential object attributes
    :func:`__repr__`: returns parameters, optimizer, scheduler, and state
    :func:`accuracy`: returns model accuracy
    :func:`compile`: assembles a softmax regression model
    :func:`find_max_batch`: computes max batch sizes for target gpu utilization
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture
    :func:`to`: move the model, optimizer, and scheduler to a device
    """

    def __init__(
        self,
        attack,
        attack_params,
        batch_size,
        epochs,
        learning_rate,
        loss,
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        auto_batch=0,
        benchmark=True,
        classes=None,
        device="cpu",
        threads=-1,
        verbosity=0.25,
    ):
        """
        This method instantiates LinearClassifier objects with attributes
        necessary to support model (adversarial) training and logging.
        Importantly, models are not usable until they are trained (via fit),
        this "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to Keras) on
        initialization.

        :param attack: attack for (optionally) performing adversarial training
        :type attack: aml Attack or Adversary class or None
        :param attack_params: attack parameters for adversarial training
        :type attack_params: dict
        :param auto_batch: find max batch size for target gpu utilization
        :type auto_batch: float
        :param batch_size: training batch size (-1 for 1 batch)
        :type batch_size: int
        :param benchmark: enable torch auto-tuner for cuDNN algorithms
        :type benchmark: bool
        :param classes: number of classes
        :type classes: int
        :param device: hardware device to use
        :type device: str
        :param epochs: number of training iterations
        :type epochs: int
        :param learning_rate: learning rate
        :type learning_rate: float
        :param loss: loss function
        :type loss: torch.nn.modules.loss class
        :param optimizer: optimizer from optimizers package
        :type optimizer: torch.optim class
        :param optimizer_params: optimizer parameters
        :type optimizer_params: dict
        :param scheduler: scheduler for dynamic learning rates
        :type scheduler: torch.optim.lr_scheduler class or None
        :param scheduler_params: scheduler parameters
        :type scheduler_params: dict
        :param threads: number of cpu threads used for training
        :type threads: int (-1 for for max threads)
        :param verbosity: print training statistics every verbosity%
        :type verbosity: float
        :return: a linear classifier skeleton
        :rtype: LinearClassifier object
        """
        self.attack_alg = attack
        self.attack_params = attack_params
        self.auto_batch = auto_batch * (device == "cuda")
        self.batch_size = batch_size
        self.benchmark = benchmark
        self.classes = classes
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_func = loss
        self.optimizer_alg = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler_alg = scheduler
        self.scheduler_params = scheduler_params
        self.threads = (
            torch.get_num_threads()
            if threads == -1
            else min(threads, torch.get_num_threads())
        )
        self.verbosity = max(int(epochs * verbosity), 1 if verbosity else 0)
        self.state = "skeleton"
        self.params = {
            "attack": "N/A" if self.attack_alg is None else self.attack_alg.__name__,
            "classes": "TBD" if self.classes is None else self.classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "features": "TBD",
            "learning_rate": self.learning_rate,
            "loss": self.loss_func.__name__,
            "optimizer": self.optimizer_alg.__name__,
            "scheduler": "N/A"
            if self.scheduler_alg is None
            else self.scheduler_alg.__name__,
        }
        return None

    def __call__(self, x, grad_enabled=True):
        """
        This method returns the model logits. Optionally, gradient-tracking can
        be disabled for fast inference. Notably, when using nvidia gpus, if
        auto_batch was nonzero on __init__, then inputs are auto-batched to
        sizes (determined by find_max_batch) that will fit within nvidia gpu
        vram prevent out-of-memory errors depending on whether we are training
        the model or performing inference.

        :param x: the batch of inputs
        :type x: torch Tensor object (n, m)
        :param grad_enabled: whether gradients are calculated
        :type grad_enabled: bool
        :return: model logits
        :rtype: torch Tensor object (n, c)
        """
        split = self.sizes["inference"] if x.is_cuda and not grad_enabled else x.size(0)
        with torch.set_grad_enabled(grad_enabled):
            return torch.cat([self.model(xb.flatten(1)) for xb in x.split(split)])

    def __getattr__(self, name):
        """
        This method ostensibly aliases torch.nn.modules.container Sequential
        objects (i.e., self.model) attributes to be accessible by this object
        directly. It is principally used for readability and debugging.

        :param name: name of the attribute to recieve from self.model
        :type name: str
        :return: the desired attribute (if it exists)
        :rtype: misc
        """
        return self.__getattribute__("model").__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of parameters,
        optimizer, scheduler, state, and hardware device.

        :return: algorithms used and hyperparameters
        :rtype: str
        """
        p = ", ".join(f"{p}={v}" for p, v in self.params.items())
        return f"{type(self).__name__}({p}, state={self.state}, device={self.device})"

    def accuracy(self, x, y, as_tensor=False):
        """
        This method returns the fraction of inputs classified correctly over
        the total number of samples (unless as_tensor is True, which returns a
        boolean tensor instead).

        :param x: batch of inputs
        :type x: torch Tensor object (n, m)
        :param y: batch of labels
        :type y: Pytorch Tensor object (n,)
        :param as_tensor: whether to return a boolean, instead of a float, tensor
        :type as_tensor: bool
        :return: model accuracy
        :rtype: torch Tensor object (1,)
        """
        t = self(x, grad_enabled=False).argmax(1).eq_(y)
        return t.to(dtype=torch.bool) if as_tensor else t.mean(dtype=torch.float)

    def compile(self):
        """
        This method instantiates a torch.nn.modules.container Sequential
        object. This abstraction allows us to dynamically build models based on
        the passed-in dataset.

        :return: an untrained linear classifier
        :rtype: torch.nn.modules.container Sequential object
        """
        return torch.nn.Sequential(torch.nn.LazyLinear(self.classes))

    def find_max_batch(self, target, lower=1, upper=500000):
        """
        This method computes the maximum possible batch size for a given target
        utilization without causing out-of-memory errors in gpus for three
        scenarios: (1) model training, (2) inference, and (3) crafting
        adversarial examples. The optimal batch size across these scenarios is
        determined via binary search. Notably, this method should only be
        called when model state is not "skeleton" (i.e, the PyTorch sequential
        container has been instantiated and its lazy modules initialized).

        :param target: target memory utilization
        :type target: float
        :param lower_batch: smallest batch size to consider
        :type lower_batch: int
        :param upper_batch: largest batch size to consider
        :type upper_batch: int
        :return: max batch sizes for traing, inference, and crafting
        :rtype: dict of ints
        """

        # return unconstrained batch sizes if target utlization is 0%
        stages = "training", "inference", "crafting"
        if target == 0:
            return dict.fromkeys(stages, torch.iinfo(torch.int).max)

        # initialize stage tracking parameters
        sizes = {s: 1 for s in stages}
        utils = {s: 0 for s in stages}
        steps = torch.tensor(upper).log2().add(1).int()
        feat = self.params["features"]
        _, total_memory = torch.cuda.mem_get_info()

        # update stage state and track grads appropriately
        for stage in stages:
            self.model.requires_grad_(stage == "training")
            low = lower
            up = upper
            for i in range(steps):
                print(f"Testing {stage} batch sizes... {i / steps:.2%}", end="\r")
                try:
                    # compute utilization
                    size = (low + up) // 2
                    batch = torch.empty(
                        (size, feat),
                        device=self.device,
                        requires_grad=stage == "crafting",
                    )
                    out = self.model(batch)
                    stage != "inference" and out.sum().backward()
                    util = torch.cuda.memory_reserved() / total_memory

                    # adjust next size and save closest size under target
                    if util < target:
                        low = max(low, size)
                        sizes[stage] = max(sizes[stage], size)
                        utils[stage] = max(utils[stage], util)
                    else:
                        raise RuntimeError

                # gpu was oversubscribed; decrease batch size
                except RuntimeError:
                    up = min(up, size)

                # release resources
                out = None
                batch.grad = None
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        r = (f"{stage}: {sizes[stage]} ({utils[stage]:.2%})," for stage in stages)
        print("Batch size search complete.", *r, end="\n" if self.verbosity else "\r")
        return sizes

    def fit(self, x, y, valset=0.0):
        """
        This method is the heart of all LinearClassifier-inherited objects. It
        performs four functions: (1) instantiates a model (i.e.,
        torch.nn.modules.container Sequential object), (2) initializes training
        prerequisites (and updates metadata) (3) performs adversarial training
        (if desired), and (4) computes training (plus validation and
        robustness) statistics.

        :param x: training inputs
        :type x: torch Tensor object (n, m)
        :param y: training labels
        :type y: torch Tensor object (n,)
        :param valset: hold-out set or proportion of training data use
        :type valset: tuple of torch Tensor objects (n, m) & (n,) or float
        :param attack: attack for (optionally) performing adversarial training
        :type attack: aml Attack object or None
        :return: a trained linear classifier model
        :rtype: LinearClassifier object
        """

        # configure verbosity, compile the model, and set prerequisites
        verbose = self.verbosity and self.verbosity != self.epochs
        self.classes = y.unique().numel() if self.classes is None else self.classes
        self.model = self.compile().to(self.device)
        tsub, vsub = self.initialize(x, y, valset)
        if verbose:
            print(f"Training set: ({len(tsub)}, {x.size(1)}) × ({len(tsub)},)")
            len(vsub) > 0 and print(
                f"Validation set: ({len(vsub)}, {x.size(1)}) × ({len(vsub)},)"
            )
            self.summary()

        # configure dataloaders, results dataframe, threads, and training mode
        self.batch_size = min(self.batch_size, self.sizes["training"])
        tset = torch.utils.data.DataLoader(tsub, self.batch_size, shuffle=True)
        vset = torch.utils.data.DataLoader(vsub, max(1, len(vsub)))
        parts, stats = ("training", "validation", "adversarial"), ("accuracy", "loss")
        metrics = ["epoch"] + [f"{p}_{m}" for p in parts for m in stats]
        self.res = pandas.DataFrame(0, index=range(1, self.epochs + 1), columns=metrics)
        max_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        torch.backends.cudnn.benchmark = self.benchmark
        self.model.train()
        d = (
            torch.cuda.get_device_name()
            if x.is_cuda
            else "apple gpu"
            if x.device.type == "mps"
            else f"{self.threads} cpu threads"
        )
        print(
            f"Performing{'' if self.attack is None else ' adversarial'} training "
            f"{'' if self.attack is None else f'with {self.attack} '}"
            f"for {self.epochs} epochs on {d}...",
            end="\n" if verbose else "\r",
        )
        for e in range(1, self.epochs + 1):
            tloss = tacc = 0

            # perform one iteration of training
            for xb, yb in tset:
                self.model.requires_grad_(False)
                if self.attack is not None:
                    xb = xb.add(self.attack.craft(xb, yb, reset=True))
                self.model.requires_grad_(True)
                logits = self(xb)
                loss = self.loss(logits, yb)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.requires_grad_(False)
                tloss += loss.detach()
                tacc += logits.detach().argmax(1).eq_(yb).sum()
            self.scheduler is not None and self.scheduler.step()

            # compute training statistics every epoch and update results
            prog = self.progress(e, tacc.div(len(tsub)).item(), tloss.item(), vset)
            print(
                f"Epoch {e:{len(str(self.epochs))}} / {self.epochs} {prog}"
            ) if verbose and not e % self.verbosity else print(
                f"Epoch {e}... ({e / self.epochs:.1%})",
                end="\x1b[K\r",
            )

        # set model to eval mode, restore thread count, and update state
        self.model.eval()
        torch.set_num_threads(max_threads)
        torch.backends.cudnn.benchmark = False
        self.state = (
            "trained"
            if self.attack is None
            else f"adversarially trained ({self.attack.name})"
        )
        return self

    def initialize(self, x, y, valset):
        """
        This method initializes all objects necessary for training models and
        updates metadata. Specifically, it: (1) initializes the model with a
        small forward pass, (2) instantiates optimizers & schedulers and
        attaches them to model parameters, (3) determines the maximum batch
        size if using gpus and auto_batch is True, (4) prepares the validation
        dataset, and (5) attaches attacks for adverarial training, and (6)
        updates parameters for __repr__ and model state.

        :param x: training inputs
        :type x: torch Tensor object (n, m)
        :param y: training labels
        :type y: torch Tensor object (n,)
        :param valset: hold-out set or proportion of training data use
        :type valset: tuple of torch Tensor objects (n, m) & (n,) or float
        :return: training and validation sets
        :rtype: tuple of torch.utils.data TensorDataset objects
        """

        # initialize model and instantiate loss, optimizer, scheduler, & attack
        self.model(x[:1])
        self.loss = self.loss_func()
        self.optimizer = self.optimizer_alg(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.scheduler = (
            None
            if self.scheduler_alg is None
            else self.scheduler_alg(self.optimizer, **self.scheduler_params)
        )
        self.attack = (
            None
            if self.attack_alg is None
            else self.attack_alg(**self.attack_params | {"model": self})
        )

        # update metadata and model state
        self.params["attack"] = "N/A" if self.attack is None else repr(self.attack)
        self.params["classes"] = self.classes
        self.params["features"] = x.size(1)
        self.state = "untrained"

        # find max batch size, build validation set, and prepare data loader
        self.sizes = self.find_max_batch(self.auto_batch)
        dataset = torch.utils.data.TensorDataset(x, y)
        if isinstance(valset, float):
            nval = int(y.numel() * valset)
            idx = torch.randperm(y.numel())
            tsub = torch.utils.data.Subset(dataset, idx[nval:])
            vsub = torch.utils.data.Subset(dataset, idx[:nval])
        else:
            tsub = dataset
            vsub = torch.utils.data.TensorDataset(*valset)
        return tsub, vsub

    def load(self, path):
        """
        This method loads pretrained PyTorch model parameters. Specifically,
        this supports loading either: (1) complete models (e.g., trained torch
        Sequential containers), or (2) trained parameters (e.g., state_dict).
        Notably, either object is assumed to saved via torch.save(). Finally,
        many of the useful attributes set during fit are estimated from the
        model directly (e.g., number of features, classes, etc.), the model is
        put into inference mode, and the maximum batch size is computed if
        auto_batch is True.

        Notably, there are some subtlies in loading state dictionaries with
        lazy modules in that they *must* be initialized before the state dict
        can be loaded. To this end, if a torch.nn.modules.container Sequential
        object is not an attribute of this class, then the number of input
        features and clases are inferred from the state dict, compile is
        called, and model is initialized through a dry run.

        :param path: path to the pretrained model
        :type path: pathlib Path object
        :return: a pretrained linear classifier model
        :rtype: LinearClassifier object
        """

        # load model or state dict (and forcibly compile if necessary)
        model = torch.load(path, map_location=self.device)
        if type(model) == torch.nn.Sequential:
            self.model = model
        else:
            try:
                self.model.load_state_dict(model)
            except AttributeError:
                features = (
                    torch.tensor(self.shape).prod()
                    if hasattr(self, "shape")
                    else next(iter(model.values())).size(1)
                )
                self.classes = next(reversed(model.values())).size(0)
                self.model = self.compile()
                self(torch.empty((1, features)), grad_enabled=False)
                self.model.load_state_dict(model)
                self.model.to(self.device)

        # infer features, update parameters, and set model to inference
        features = (
            torch.tensor(self.model[0].unflattened_size).prod()
            if type(self.model[0]) is torch.nn.Unflatten
            else next(iter(self.model.state_dict().values())).size(0)
        )
        self.classes = self.model[-1].out_features
        self.params["features"] = features
        self.params["classes"] = self.classes
        self.state = "pretrained"
        self.summary()
        self.model.eval()
        return self

    def progress(self, e, tacc, tloss, vset):
        """
        This method measures various statistics on the training process and a
        formatted string concisely representing the state model. Specifically,
        this computes the following at each epoch (and the change since the
        last epoch): (1) training accuracy & loss, (2) validation accuracy &
        loss, and (3) adversarial accuracy & loss (on the validation set).

        :param e: current epoch
        :type e: int
        :param tacc: current training accuracy
        :type tacc: float
        :param tloss: current training loss
        :type tloss: float
        :param vset: validation set
        :type vset: torch.utils.data DataLoader object
        :return: print-ready statistics
        :rtype: str
        """

        # compute validation statistics and str representation
        vacc = vloss = aacc = aloss = 0
        vstr = astr = ""
        if len(vset) > 0:
            vx, vy = next(iter(vset))
            logits = self(vx, grad_enabled=False)
            vloss = self.loss(logits, vy).item()
            vacc = logits.argmax(1).eq_(vy).mean(dtype=torch.float).item()
            vstr = (
                f"Val Acc: {vacc:.1%} "
                f"({vacc - self.res.validation_accuracy.iloc[e - 2]:+.2%}) "
                f"Val Loss {vloss:.2f} "
                f"({vloss - self.res.validation_loss.iloc[e - 2]:+.2f}) "
            )

            # compute adversarial metrics and str representation
            if self.attack is not None:
                vxa = vx.add(self.attack.craft(vx, vy, reset=True))
                logits = self(vxa, grad_enabled=False)
                aloss = self.loss(logits, vy).item()
                aacc = logits.argmax(1).eq_(vy).mean(dtype=torch.float).item()
                astr = (
                    f"Adv Acc: {aacc:.1%} "
                    f"({aacc - self.res.adversarial_accuracy.iloc[e - 2]:+.2%}) "
                    f"Adv Loss: {aloss:.2f} "
                    f"({aloss - self.res.adversarial_loss.iloc[e - 2]:+.2f}) "
                )
        self.res.loc[e] = e, tacc, tloss, vacc, vloss, aacc, aloss

        # build str representation and return
        return (
            f"Accuracy: {tacc:.1%} "
            f"({tacc - self.res.training_accuracy.iloc[e - 2]:+.2%}) "
            f"Loss: {tloss:.2f} "
            f"({tloss - self.res.training_loss.iloc[e - 2]:+.2f}) "
            f"{vstr}{astr}"
        )

    def save(self, path, slim=True):
        """
        This method saves either the entire model or just the state_dict
        associated with the torch.nn.modules.container Sequential containers.

        :param path: path (including filename) to save the model
        :type path: pathlib Path object
        :param slim: whether to only save the state_dict (or the entire model)
        :type slim: bool
        :return: None
        :rtype: NoneType
        """
        torch.save(self.model.state_dict() if slim else self.model.cpu(), path)
        return None

    def summary(self):
        """
        This method prints the model architecture.

        :return: None
        :rtype: NoneType
        """
        print(f"Defined model:\n{self.model}")
        return None

    def to(self, device):
        """
        This method moves the model, optimizer, and scheduler to the cpu, an
        nvidia gpu, or a macOS gpu. At this time, device conversions are not
        supported for optimizers nor schedulers
        (https://github.com/pytorch/pytorch/issues/41839), so we leverage a
        trick shown in https://github.com/pytorch/pytorch/issues/8741 to
        refresh the state of optimizers and schedulers (as these subroutines
        match device state to that of the attached parameters).

        :param device: device to switch to
        :type device: str
        :return: None
        :rtype: NoneType
        """
        self.device = device
        self.model.to(self.device)

        # models in pretrained state may not have an optimizer nor scheduler
        if self.state != "pretrained":
            self.optimizer.load_state_dict(self.optimizer.state_dict())
            self.scheduler_alg is not None and self.scheduler.load_state_dict(
                self.scheduler.state_dict()
            )
        return None


class MLPClassifier(LinearClassifier):
    """
    This class extends the LinearClassifier class via dropout, hidden_layers
    and activation arguments. Specifically, a MLPClassifier object is a
    PyTorch-based multi-linear-layer classification model (i.e., a multi-layer
    perceptron). This class inherits the following methods as-is from the
    LinearClassifier class:

    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch.nn Sequential object attributes
    :func:`__repr__`: returns parameters, optimizer, scheduler, and state
    :func:`accuracy`: returns model accuracy
    :func:`find_max_batch`: computes max batch sizes for target gpu utilization
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture
    :func:`to`: move the model, optimizer, and scheduler to a device

    And redefines the following methods:

    :func:`__init__`: instantiates MLPClassifier objects
    :func:`compile`: assembles an multi-layer perceptron model
    """

    def __init__(
        self,
        activation,
        dropout,
        hidden_layers,
        **lc_args,
    ):
        """
        This method instantiates MLPClassifier objects with the attributes
        necessary to support model (adversarial) training and logging.
        Importantly, models are not usable until they are trained (via fit).
        This "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to Keras) on
        initialization.

        :param activation: activation function at each hidden layer
        :type activation: torch.nn.modules.activation class
        :param dropout: dropout probability
        :type dropout: float
        :param hidden_layers: the number of neurons at each hidden layer
        :type hidden_layers: tuple of ints
        :param lc_args: LinearClassifier arguments
        :type lc_args: dict
        :return: a mutli-layer linear classifier skeleton
        :rtype: MLPClassifier object
        """
        super().__init__(**lc_args)
        self.activation = activation
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.params = self.params | {
            "activation": activation.__name__,
            "dropout": dropout,
            "hidden_layers": hidden_layers,
        }
        return None

    def compile(self):
        """
        Like the implementation of compile from the LinearClassifier class,
        this method instantiates a torch.nn.modules.container Sequential
        object. It additionally adds support for interleaving activation
        functions, dropout layers, and hidden layers. This abstraction allows
        us to dynamically build models based on the passed-in dataset.

        :return: an untrained multi-layer linear classifier
        :rtype: torch.nn.modules.container Sequential object
        """
        components = (
            [torch.nn.Dropout(self.dropout)],
            map(torch.nn.LazyLinear, self.hidden_layers),
            [self.activation()],
        )
        return torch.nn.Sequential(
            *itertools.chain(*itertools.product(*components)), *super().compile()
        )


class CNNClassifier(MLPClassifier):
    """
    This class extends the MLPClassifier class via conv_layers and kernel_size.
    Specifically, a CNNClassifier object is a PyTorch-based
    multi-convolutional-and-linear-layer classification model (i.e., a
    convolutional neural network). This class inherits the following methods
    as-is from the MLPClassifier class:

    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch.nn Sequential object attributes
    :func:`__repr__`: returns parameters, optimizer, scheduler, and state
    :func:`accuracy`: returns model accuracy
    :func:`find_max_batch`: computes max batch sizes for target gpu utilization
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture
    :func:`to`: move the model, optimizer, and scheduler to a device

    And redefines the following methods:

    :func:`__init__`: instantiates CNNClassifier objects
    :func:`compile`: assembles a convolutional neural network
    """

    def __init__(
        self,
        conv_layers,
        kernel_size,
        shape,
        **mlp_args,
    ):
        """
        This method instantiates CNNClassifier objects with the attributes
        necessary to support model (adversarial) training and logging.
        Importantly, models are not usable until they are trained (via fit).
        This "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to Keras) on
        initialization.

        :param conv_layers: the number of filters at each convolutional layer
        :type conv_layers: tuple of ints
        :param kernel_size: size of the convolving kernel
        :type kernel_size: int
        :param shape: original input shape (channels, width, height)
        :type shape: tuple of ints
        :param mlp_args: MLPClassifier arguments
        :type mlp_args: dict
        :return: a convolutional neural network classifier skeleton
        :rtype: CNNClassifier object
        """
        super().__init__(**mlp_args)
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.shape = shape
        self.params = self.params | {
            "conv_layers": conv_layers,
            "kernel_size": kernel_size,
        }
        return None

    def compile(self):
        """
        Like the implementation of compile from the MLPClassifier class, this
        method instantiates a torch.nn.modules.container Sequential object. It
        additionally adds support for interleaving activation functions,
        convolutional layers, and maxpool layers. This abstraction allows us to
        dynamically build models based on the passed-in dataset. Importantly,
        this method asssumes that inputs are flattened vectors to be reshaped
        according to shape for interoperability with
        https://github.com/sheatsley/datasets.

        :return: an untrained convolutional neural network classifier
        :rtype: torch.nn.modules.container Sequential object
        """

        # assemble convolutional layers
        convolutional = (
            map(lambda c: torch.nn.LazyConv2d(c, self.kernel_size), self.conv_layers),
            [self.activation()],
            [torch.nn.MaxPool2d(self.kernel_size)],
        )

        # unflatten and attach convolutional and linear layers with flatten
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, self.shape),
            *itertools.chain(*itertools.product(*convolutional)),
            torch.nn.Flatten(),
            *super().compile(),
        )

"""
This module defines the core for the deep learning models repo. All supported
deep learning models are defined here, with the LinearClassifier class defining
the primary features across all models (which all inherit from
LinearClassifier). Much like Keras, these classes define simple interfaces to
support rapid prototyping of standard deep learning models.
Author: Ryan Sheatsley & Blaine Hoak
Thu Feb 2 2023
"""
import dlm.utilities as utilities  # miscellaneous utility functions
import itertools  # Functions creating iterators for efficient looping
import pandas  # Python Data Analysis Library
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add an examples directory showing plotting, etc.
# update all hyperparameters
# remove defaults
# add trades loss
# add mart loss
# add cifar10 hparams
# remove hparam unittest
# change hparam named tuple to hparams instead of datasets (aml calls it "templates")
# figure out chained warning thing
# change template to hyperparameters
# redo repr
# create thread benchmarks
# support checkpoints
# cleanup compile (pretty ugly)


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
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`max_batch_size`: determines max batch size for gpus
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture
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
        auto_batch=True,
        classes=None,
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
        :param auto_batch: determine max batch size for gpu vram
        :type auto_batch: bool
        :param classes: number of classes
        :type classes: int
        :param batch_size: training batch size (-1 for 1 batch)
        :type batch_size: int
        :param epochs: number of training iterations
        :type epochs: int
        :param learning_rate: learning rate
        :type learning_rate: float
        :param loss: loss function
        :type loss: loss module class
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
        self.auto_batch = auto_batch
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_func = loss
        self.optimizer_alg = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler_alg = scheduler
        self.schedular_params = scheduler_params
        self.threads = (
            torch.get_num_threads()
            if threads == -1
            else min(threads, torch.get_num_threads())
        )
        self.verbosity = max(int(epochs * verbosity), 1 if verbosity else 0)
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
            "state": "skeleton",
        }
        return None

    def __call__(self, x, grad_enabled=True):
        """
        This method returns the model logits. Optionally, gradient-tracking can
        be disabled for fast inference. Notably, when using gpus, if auto_batch
        was set to True on __init__, then inputs are auto-batched to sizes
        (determined by max_batch_size) that will fit within GPU VRAM to prevent
        out-of-memory errors.

        :param x: the batch of inputs
        :type x: torch Tensor object (n, m)
        :param grad_enabled: whether gradients are calculated
        :type grad_enabled: bool
        :return: model logits
        :rtype: torch Tensor object (n, c)
        """
        with torch.set_grad_enabled(grad_enabled):
            xb = x.split(x.size(0) if self.max_batch is None else self.max_batch)
            return torch.cat([self.model(x) for x in xb])

    def __getattr__(self, name):
        """
        This method ostensibly aliases torch.nn Sequential objects (i.e.,
        self.model) attributes to be accessible by this object directly. It is
        principally used for better readability and easier debugging.

        :param name: name of the attribute to recieve from self.model
        :type name: str
        :return: the desired attribute (if it exists)
        :rtype: misc
        """
        return self.__getattribute__("model").__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of parameters,
        optimizer, scheduler, and state.

        :return: algorithms used and hyperparameters
        :rtype: str
        """
        params = (f"{p}={v}" for p, v in self.params.items())
        return f"{type(self).__name__}({', '.join(params)})"

    def accuracy(self, x, y):
        """
        This method returns the fraction of inputs classified correctly over
        the total number of samples.

        :param x: batch of inputs
        :type x: torch Tensor object (n, m)
        :param y: batch of labels
        :type y: Pytorch Tensor object (n,)
        :return: model accuracy
        :rtype: torch Tensor object (1,)
        """
        return self(x, grad_enabled=False).argmax(1).eq_(y).mean(dtype=torch.float)

    def compile(self):
        """
        This method instantiates a torch.nn Sequential object. This abstraction
        allows us to dynamically build models based on the passed-in dataset.

        :return: an untrained linear classifier
        :rtype: torch.nn Sequential object
        """
        return torch.nn.Sequential(torch.nn.LazyLinear(self.classes))

    def cpu(self):
        """
        This method moves the model, optimizer, and scheduler to the cpu. At
        this time, to and cpu methods are not supported for optimizers nor
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
        scheduler states. Given that this process is device agnostic (in that
        the states are simply refreshed), it is expected that method be called
        by either cpu() or cuda().

        :return: None
        :rtype: NoneType
        """
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        self.scheduler_alg is not None and self.scheduler.load_state_dict(
            self.scheduler.state_dict()
        )
        return None

    def fit(self, x, y, valset=0.0):
        """
        This method is the heart of all LinearClassifier-inherited objects. It
        performs four functions: (1) instantiates a model (i.e., torch.nn
        Sequential object), (2) initializes training prequisites (and updates
        metadata) (3) optionally performs adversarial training, and (4)
        computes training (plus validation and robustness) statistics.

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

        # compile the model and initialize training prerequisites
        self.classes = y.unique().numel() if self.classes is None else self.classes
        self.model = self.compile()
        tset, vset = self.initialize(x, y, valset)
        self.summary()

        # configure verbosity, results dataframe, threads, and training mode
        verbose = self.verbosity and self.verbosity != self.epochs
        parts, stats = ("training", "validation", "adversarial"), ("accuracy", "loss")
        metrics = ["epoch"] + [f"{p}_{m}" for p in parts for m in stats]
        self.res = pandas.DataFrame(0, index=range(1, self.epochs + 1), columns=metrics)
        max_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        self.model.train()
        d = torch.cuda.get_device_name() if x.is_cuda else f"cpu ({self.threads})"
        print(
            f"Performing {'' if self.attack is None else 'adversarial'} training",
            f"{'' if self.attack is None else f'with {self.attack}'}",
            f"for {self.epochs} epochs on {d}...",
        )

        # enter main training loop; apply scheduler and reset epoch loss
        for e in range(1, self.epochs + 1):
            self.scheduler is not None and self.scheduler.step()
            tloss = tacc = 0

            # perform one iteration of training
            for xb, yb in tset:
                self.model.requires_grad_(False)
                if self.attack is not None:
                    xb = self.attack.craft(xb, yb, reset=True)
                self.model.requires_grad_(True)
                logits = self(xb)
                loss = self.loss(logits, yb)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.requires_grad_(False)
                tloss += loss.detach()
                tacc += logits.detach().argmax(1).eq_(yb).sum()

            # compute training statistics every epoch and update results
            prog = self.progress(tacc.mean().item(), tloss.item(), vset)
            print(
                f"Epoch {e:{len(str(self.epochs))}} / {self.epochs} {prog}"
            ) if verbose and not e % self.verbosity else print(
                f"Epoch {e}... ({e / self.epochs:.1%})",
                end="\r",
            )

        # set model to eval mode, restore thread count, and update state
        self.model.eval()
        torch.set_num_threads(max_threads)
        self.params["state"] = (
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
        :rtype: tuple of torch.utils.data DataLoader objects
        """

        # initialize model and instantiate optimizer, scheduler, & attack
        self.model(x[:1])
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
            else self.attack_alg(**self.attack_params | {"model": self.model})
        )

        # find max batch size, build validation set, and prepare data loader
        self.auto_batch and self.max_batch_size()
        dataset = torch.utils.data.TensorDataset(x, y)
        if isinstance(valset, float):
            nval = int(y.numel() * valset)
            idx = torch.randperm(y.numel())
            tsub = torch.utils.data.Subset(dataset, idx[nval:])
            vsub = torch.utils.data.Subset(dataset, idx[:nval])
            ntrain = y.numel() - nval
        else:
            tsub = dataset
            vsub = torch.utils.data.TensorDataset(*valset)
            nval = len(vsub)
            ntrain = len(tsub)
        tset = torch.utils.data.DataLoader(tsub, self.batch_size, shuffle=True)
        vset = torch.utils.data.DataLoader(vsub, len(vsub))
        print(f"Training set: ({ntrain}, {x.size(1)}) × ({ntrain},)")
        len(vsub) != 0 and print(f"Validation set: ({nval}, {x.size(1)}) × ({nval},)")

        # update metadata and model state
        self.params["attack"] = "N/A" if self.attack is None else repr(self.attack)
        self.params["classes"] = self.classes
        self.params["features"] = x.size(1)
        self.params["state"] = "untrained"
        return tset, vset

    def max_batch_size(self, grad=False, lower=1, upper=500000, utilization=0.95):
        """
        This method computes the maximum batch size usable for forward-only
        (i.e., inference) and forward-backward (i.e., training) passes on gpus.
        Specifically, this provides an abstraction so that large batch sizes
        that would produce out-of-memory exceptions are automatically corrected
        to smaller batches while maintaining maximum gpu utilization. This
        method applies binary search on the batch size until the memory usage
        is within the paramertized utilization. Afterwards, grad_batch and
        nograd_batch attributes are exposed (and used appropriately in __call__
        based on the value of grad). Notably, this method cannot be called when
        model state is "skeleton."

        :param grad: find sizes for forward-backward only or including forward
        :type grad: bool
        :param lower_batch: smallest batch size to consider
        :type lower_batch: int
        :param stages: find forward-only (0) and/or forward-backward (1) sizes
        :type stages: list of ints
        :param upper_batch: largest batch size to consider
        :type upper_batch: int
        :param utilization: minimum percentage of memory to be used
        :type utilization: float
        :return: max batch sizes for forward-only and forward-backward passes
        :rtype: tuple of ints
        """
        features = self.params["features"]
        stage_names = ("forward-pass-only", "forward-backward-passes")
        stage = stage_names[grad]
        free_memory, max_memory = torch.cuda.mem_get_info()
        print(f"Estimating {stage} maximum batch size...")
        try:

            # compute utilization
            batch_size = (upper - lower) // 2
            out = self.model(torch.empty((batch_size, features)), grad)
            out.sum().backward() if grad else None
            curr_util = torch.cuda.memory_allocated() / max_memory
            print(f"{stage} utilization (batch size={batch_size}: {curr_util}")

            # utilization target was met
            if curr_util >= utilization:
                print(f"{stage} utilization target met ({curr_util:.1%}!")

                # return forward-backward (keep forward size if found)
                return (batch_size,) * 2 if grad else batch_size, self.max_batch_size(
                    True, 1, batch_size, utilization
                )[1]

            # required utilization is too high (search failed); drop by 5%
            elif not batch_size:
                utilization -= 0.05
                print(f"Utilization is too strict! Dropping to {utilization:.0%}...")
                return self.max_batch_size(grad, 1, 500000, utilization)

            # utilization was below threshold;
            return self.max_batch_size(grad, batch_size, upper, utilization)

        # utilization was oversubscribed; cut batch size in half
        except RuntimeError:
            print(f"Out of memory for {stage} (batch_size={batch_size})!")
            self.model.zero_grad(True)
            torch.cuda.empty_cache()
            return self.max_batch_size(grad, lower, batch_size, utilization)

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
            vx, vy = vset
            logits = self(vx, grad_enabled=False)
            vloss = self.loss(logits, vy).item()
            vacc = logits.argmax(1).eq_(vy).mean().item()
            vstr = (
                f"Validation Acc: {vacc:.1%}",
                f"({vacc - self.res.validation_accuracy.iloc[-2]:+6.2%})",
                f"Validation Loss {vloss:.2f}"
                f"({vloss - self.res.validation_loss.iloc[-2]:+6.2f}) ",
            )

            # compute adversarial metrics and str representation
            if self.attack is not None:
                logits = self(self.attack.craft(vx, vy, reset=True), grad_enabled=False)
                aloss = self.loss(logits, vy).item()
                aacc = logits.argmax(1).eq_(vy).mean().item()
                astr = (
                    f"Adversarial Acc: {aacc:.1%}",
                    f"({aacc - self.res.adversarial_accuracy.iloc[-2]:+6.2%})",
                    f"Adversarial Loss: {aloss:.2f}",
                    f"({aloss - self.res.adversarial_loss.iloc[-2]:+6.2f})",
                )
        self.res.loc[e] = e, tacc, tloss, vacc, vloss, aacc, aloss

        # build str representation and return
        return (
            f"Accuracy: {tacc:.1%}",
            f"({tacc - self.res.training_accuracy.iloc[-2]:+6.2%})",
            f"Loss: {tloss:.2f}",
            f"({tloss - self.res.training_loss.iloc[-2]:+6.2f})",
            vstr,
            astr,
        )

    def load(self, path):
        """
        This method loads pretrained PyTorch model parameters. Specifically,
        this supports loading either: (1) complete models (e.g., trained torch
        Sequential containers), or (2) trained parameters (e.g., state_dict).
        Notably, either object is assumed to saved via torch.save(). Finally,
        many of the useful attributes set during fit are estimated from the
        model directly (e.g., number of features, classes, etc.) and the
        maximum batch size is computed if auto_batch is True.

        :param path: path to the pretrained model
        :type path: pathlib Path object
        :return: a pretrained linear classifier model
        :rtype: LinearClassifier object
        """
        model = torch.load(path)
        try:
            self.model.load_state_dict(model)
        except TypeError:
            self.model = model
        self.params["features"] = model[0].in_features
        self.params["classes"] = model[-1].out_features
        self.params["state"] = "pretrained"
        self.summary()
        self.auto_batch and self.max_batch_size()
        return self

    def save(self, path, slim=True):
        """
        This method saves either the entire model or just the state_dict
        associated with the torch.nn Sequential containers.

        :param path: path (including filename) to save the model
        :type path: pathlib Path object
        :param slim: whether to only save the state_dict (or the entire model)
        :type slim: bool
        :return: None
        :rtype: NoneType
        """
        torch.save(self.model.state_dict() if slim else self.model, path)
        return None

    def summary(self):
        """
        This method prints the model architecture.

        :return: None
        :rtype: NoneType
        """
        print(f"Defined model:\n{self.model}")
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
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`max_batch_size`: determines max batch size for gpus
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture

    And redefines the following methods:

    :func:`__init__`: instantiates MLPClassifier objects
    :func:`compile`: assembles an multi-layer perceptron model
    """

    def __init__(
        self,
        activation,
        dropout,
        hidden_layers,
        shape,
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
        :type activation: torch.nn class
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
        this method instantiates a torch.nn Sequential object. It additionally
        adds support for interleaving activation functions, dropout layers, and
        hidden layers. This abstraction allows us to dynamically build models
        based on the passed-in dataset.

        :return: an untrained multi-layer linear classifier
        :rtype: torch.nn Sequential object
        """
        return torch.nn.Sequential(
            *itertools.chain.from_iterable(
                zip(
                    (torch.nn.Dropout(self.dropout),) * len(self.hidden_layers),
                    map(torch.nn.LazyLinear, self.hidden_layers),
                    (self.activation(),) * len(self.hidden_layers),
                )
            ),
            torch.nn.LazyLinear(self.classes),
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
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training
    :func:`init`: initializes training preqrequisites
    :func:`load`: load saved models
    :func:`max_batch_size`: determines max batch size for gpus
    :func:`progress`: records various statistics on training progress
    :func:`save`: save models parameters (or entire model state)
    :func:`summary`: prints model architecture

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
        method instantiates a torch.nn Sequential object. It additionally adds
        support for interleaving activation functions, convolutional layers,
        and maxpool layers. This abstraction allows us to dynamically build
        models based on the passed-in dataset. Importantly, this method
        asssumes that inputs are flattened vectors to be reshaped according to
        shape for interoperability with https://github.com/sheatsley/datasets.

        :return: an untrained convolutional neural network classifier
        :rtype: torch Sequential object
        """

        # assemble convolutional and linear layers
        convolutional = itertools.chain.from_iterable(
            zip(
                map(
                    torch.nn.LazyConv2d,
                    self.conv_layers,
                    itertools.repeat(self.kernel_size),
                ),
                (self.activation(),) * len(self.conv_layers),
                (torch.nn.MaxPool2d(self.kernel_size),) * len(self.conv_layers),
            )
        )
        linear = super().compile()

        # attach unflatten and meld convolutional and linear layers with flatten
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, self.shape),
            *convolutional,
            torch.nn.Flatten(),
            *linear,
        )

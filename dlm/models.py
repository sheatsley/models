"""
This module defines the core for the deep learning models repo. All
supported deep learning models are defined here, with the LinearClassifier
class defining the primary features across all models (which all inherit from
LinearClassifier). Much like scikit-learn, these classes define simple
interfaces to support rapid prototyping of standard deep learning models.
Author: Ryan Sheatsley & Blaine Hoak
Mon Oct 24 2022
"""
import dlm.utilities as utilities  # miscellaneous utility functions
import itertools  # Functions creating iterators for efficient looping
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration

# TODO
# add an examples directory showing plotting, etc.


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
    :func:`build`: assembles a softmax regression model
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training
    :func:`max_batch_size`: determines max batch size for gpus
    :func:`prefit`: load pretrained parameters
    """

    def __init__(
        self,
        batch_size=128,
        iters=10,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        safe_batch=True,
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
        :param safe_batch: prevent gpu oom errors for large batches
        :type safe_batch: bool
        :param scheduler: scheduler for dynamic learning rates
        :type scheduler: torch optim.lr_scheduler class or None
        :param scheduler_params: scheduler parameters
        :type scheduler_params: dict
        :param threads: number of cpu threads used for training
        :type threads: int or None (for max threads)
        :param verbosity: print training statistics every verbosity%
        :type verbosity: float
        :return: a linear classifier skeleton
        :rtype: LinearClassifier object
        """
        super().__init__()
        self.batch_size = batch_size
        self.iters = iters
        self.learning_rate = learning_rate
        self.loss = loss(reduction="sum")
        self.optimizer_alg = optimizer
        self.optimizer_params = optimizer_params
        self.safe_batch = safe_batch
        self.scheduler_alg = scheduler
        self.schedular_params = scheduler_params
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
            "optimizer": optimizer.__name__,
            "state": "skeleton",
        } | ({"lr_scheduler": scheduler.__name__} if scheduler else {})
        return None

    def __call__(self, x, grad=True):
        """
        This method returns the model logits. Optionally, gradient-tracking can
        be disabled for fast inference. Importantly, when using gpus, inputs
        are auto-batched (via max_batch_size()) to sizes that will fit within
        VRAM to prevent out-of-memory errors.

        :param x: the batch of inputs
        :type x: torch Tensor object (n, m)
        :param grad: whether gradients are computed
        :type grad: bool
        :return: model logits
        :rtype: torch Tensor object (n, c)
        """
        with torch.set_grad_enabled(grad):
            xbatch = x.split(self.max_bs[grad] if self.max_bs else x.size(0))
            return torch.cat([self.model(xb.flatten(1)) for xb in xbatch])

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
        return self.__getattribute__("model").__getattribute__(name)

    def __repr__(self):
        """
        This method returns a concise string representation of algorithms,
        hyperparameters, and architecture.

        :return: algorithms used and hyperparameters
        :rtype: str
        """
        params = (f"{p}={v}" for p, v in self.params.items())
        return f"{self.__class__.__name__}({', '.join(params)})"

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
        return self(x, False).argmax(1).eq(y).sum().div(y.numel())

    def build(self, x, y):
        """
        This method instantiates a torch Sequential object. This abstraction
        allows us to dynamically build models based on the passed-in dataset,
        versus hardcoding model architectures via a forward() method.

        :param x: dataset of inputs
        :type x: torch Tensor object (n, m)
        :param y: dataset of labels
        :type y: Pytorch Tensor object (n,)
        :return: an untrained linear classifier
        :rtype: torch Sequential object
        """
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

    def fit(self, x, y, validation=0.0, atk=None, shape=None):
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
        :param shape: original shape of x (channels, width, height)
        :type shape: tuple of ints
        :return: a trained linear classifier model
        :rtype: LinearClassifier object
        """

        # instantiate model (via one forward pass), optimizer, and scheduler
        self.params["features"] = x.size(1)
        self.params["classes"] = y.unique().numel()
        self.params["state"] = "untrained"
        self.model = self.build(x, y, shape) if shape else self.build(x, y)
        self.model(x[:1])
        self.optimizer = self.optimizer_alg(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.scheduler = (
            self.scheduler_alg(self.optimizer, **self.scheduler_params)
            if self.scheduler_alg
            else None
        )
        print(f"Defined model:\n{self.model}")

        # find max batch size, build validation set, and prepare data loader
        self.max_bs = self.max_batch_size() if self.safe_batch and x.is_cuda else None
        if isinstance(validation, float):
            num_val = int(y.numel() * validation)
            perm_idx = torch.randperm(y.numel())
            x_val, x = x[perm_idx].tensor_split([num_val])
            y_val, y = y[perm_idx].tensor_split([num_val])
        else:
            x_val, y_val = validation
        print(
            f"Validation set created with shape: {x_val.size()} × {y_val.size()}"
        ) if validation else None
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
            if next(self.model.parameters()).is_cuda
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
            iter_acc = torch.tensor(0.0)

            # perform one iteration of training
            for xb, yb in trainset:
                self.model.requires_grad_(False)
                with utilities.supress_stdout(not (current_iter + 1) % self.verbosity):
                    xb = self.atk.craft(xb, yb) if atk else xb
                self.model.requires_grad_(True)
                batch_logits = self(xb)
                batch_loss = self.loss(batch_logits, yb)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                iter_loss += batch_loss.item()
                iter_acc += batch_logits.argmax(1).eq(yb).sum()

            # collect learning statistics every iteration
            self.stats["train_loss"].append(iter_loss)
            self.stats["train_acc"].append(iter_acc.div_(y.numel()).item())
            if validation:
                val_logits = self(x_val, False)
                val_loss = self.loss(val_logits, y_val).item()
                val_acc = val_logits.argmax(1).eq(y_val).sum().div(y_val.numel()).item()
                self.stats["val_loss"].append(val_loss)
                self.stats["val_acc"].append(val_acc)

            # print learning statistics every verbosity%
            print(
                f"Iter: {current_iter + 1:{len(str(self.iters))}}/{self.iters}",
                f"Loss: {iter_loss:.2f}",
                f"({iter_loss - sum(self.stats['train_loss'][-2:-1]):+6.2f})",
                f"Accuracy: {iter_acc:.1%}",
                f"({iter_acc - sum(self.stats['train_acc'][-2:-1]):+6.1%})",
                *(
                    f"Validation Loss: {self.stats['val_loss'][-1]:.2f}",
                    f"({val_loss - sum(self.stats['val_loss'][-2:-1]):+6.2f})",
                    f"Validation Acc: {self.stats['val_acc'][-1]:.1%}",
                    f"({val_acc - sum(self.stats['val_acc'][-2:-1]):+6.1%})",
                )
                if validation
                else "",
            ) if not (current_iter + 1) % self.verbosity else None

        # disable gradients, restore thread count, set state, and compute attack stats
        print(
            "Disabling gradients",
            f"and setting max threads to {max_threads}..."
            if not next(self.model.parameters()).is_cuda
            else "...",
        )
        self.model.requires_grad_(False)
        torch.set_num_threads(max_threads)
        self.params["state"] = (
            f"adversarially trained ({atk.name})" if atk else "trained"
        )
        self.model.eval()
        if atk:
            atk_logits = self.model(self.atk.craft(x, y))
            atk_loss = f"{self.loss(atk_logits, y).item():.3f}"
            atk_acc = f"{atk_logits.argmax(1).eq(y).sum().div(y.numel()).item():.1%}"
            print("Adversarial Loss:", atk_loss, "Adversarial Acc:", atk_acc)
        return self

    def max_batch_size(self, grad=False, lower=1, upper=500000, utilization=0.95):
        """
        This method computes the maximum batch size usable for forward-only
        (i.e., inference) and forward-backward (i.e., training) passes on gpus.
        Specifically, this provides an abstraction so that large batch sizes
        that would produce out-of-memory exceptions are automatically corrected
        to smaller batches while maintaining maximum gpu utilization. This
        method applies binary search on the batch size until the memory usage
        is within the paramertized utilization. Afterwards, grad_batch and
        nograd_batch attributes are exposed (and used appropriately in
        __call__() based on the value of grad). Notably, this method cannot be
        called when model state is "skeleton."

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

    def prefit(self, path):
        """
        This method loads pretrained PyTorch model parameters. Specifically,
        this supports loading either: (1) complete models (e.g., trained torch
        Sequential containers), or (2) trained parameters (e.g., state_dict's).
        Notably, either object is assumed to saved via torch.save(). Finally,
        many of the useful attributes set during fit() are estimated from the
        model directly (e.g., number of features, classes, etc.)

        :param path: path to the pretrained model
        :type path: pathlib Path object
        :return: a pretrained linear classifier model
        :rtype: LinearClassifier object
        """
        # load parameters, set attributes, and find max batch size (if on gpu)
        model = torch.load(path)
        try:
            self.model.load_state_dict(model)
        except TypeError:
            self.model = model
        self.params["features"] = model[0].in_features
        self.params["classes"] = model[-1].out_features
        self.params["state"] = "pretrained"
        print(f"Defined (pretrained) model:\n{self.model}")
        self.max_batch_size() if self.safe_batch and next(
            model.parameters()
        ).is_cuda else None
        return self


class MLPClassifier(LinearClassifier):
    """
    This class extends the LinearClassifier class via dropout, hidden_layers
    and activation arguments. Specifically, a MLPClassifier object is a
    PyTorch-based multi-linear-layer classification model (i.e., a multi-layer
    perceptron). This class inherits the following methods as-is from the
    LinearClassifier class:

    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch nn.Sequential object attributes
    :func:`__repr__`: returns architecture, hyperparameters, and algorithms
    :func:`accuracy`: returns model accuracy
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training

    And redefines the following methods:

    :func:`__init__`: instantiates MLPClassifier objects
    :func:`build`: assembles an multi-layer perceptron model
    """

    def __init__(
        self,
        activation=torch.nn.ReLU,
        batch_size=128,
        dropout=0.0,
        hidden_layers=(15,),
        iters=10,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        safe_batch=True,
        scheduler=None,
        scheduler_params={},
        threads=None,
        verbosity=0.25,
    ):
        """
        This method instantiates MLPClassifier objects with a variety of
        attributes necessary to support model training and logging.
        Importantly, models are not usable until they are trained (via fit()),
        this "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to scikit-learn
        model classes) on initialization.

        :param activation: activation function at each hidden layer
        :type activation: torch nn class
        :param batch_size: training batch size (-1 for 1 batch)
        :type batch_size: int
        :param dropout: dropout probability
        :type dropout: float
        :param hidden_layers: the number of neurons at each hidden layer
        :type hidden_layers: tuple of ints
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
        :param safe_batch: prevent gpu oom errors for large batches
        :type safe_batch: bool
        :param scheduler: scheduler for dynamic learning rates
        :type scheduler: torch optim.lr_scheduler class or None
        :param scheduler_params: scheduler parameters
        :type scheduler_params: dict
        :param threads: number of cpu threads used for training
        :type threads: int or None (for max threads)
        :param verbosity: print training statistics every verbosity%
        :type verbosity: float
        :return: a mutli-layer linear classifier skeleton
        :rtype: MLPClassifier object
        """
        super().__init__(
            batch_size,
            iters,
            learning_rate,
            loss,
            optimizer,
            optimizer_params,
            safe_batch,
            scheduler,
            scheduler_params,
            threads,
            verbosity,
        )
        self.activation = activation
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.params.update(
            {
                "activation": activation.__name__,
                "dropout": dropout,
                "hidden_layers": hidden_layers,
            }
        )
        return None

    def build(self, x, y):
        """
        Like the implementation of build() from the LinearClassifier class,
        this method instantiates a torch Sequential object. It additionally
        adds support for interleaving activation functions, dropout layers, and
        hidden layers. This abstraction allows us to dynamically build models
        based on the passed-in dataset, versus hardcoding model architectures
        via a forward() method.

        :param x: dataset of inputs
        :type x: torch Tensor object (n, m)
        :param y: dataset of labels
        :type y: Pytorch Tensor object (n,)
        :return: an untrained multi-layer linear classifier
        :rtype: torch Sequential object
        """
        return torch.nn.Sequential(
            *itertools.chain.from_iterable(
                zip(
                    (torch.nn.Dropout(self.dropout),) * len(self.hidden_layers),
                    map(torch.nn.LazyLinear, self.hidden_layers),
                    (self.activation(),) * len(self.hidden_layers),
                )
            ),
            torch.nn.LazyLinear(y.unique().numel()),
        )


class CNNClassifier(MLPClassifier):
    """
    This class extends the MLPClassifier class via conv_layers and kernel_size.
    Specifically, a CNNClassifier object is a PyTorch-based
    multi-convolutional-and-linear-layer classification model (i.e., a
    convolutional neural network). This class inherits the following methods
    as-is from the MLPClassifier class:

    :func:`__call__`: returns model logits
    :func:`__getattr__`: return torch nn.Sequential object attributes
    :func:`__repr__`: returns architecture, hyperparameters, and algorithms
    :func:`accuracy`: returns model accuracy
    :func:`cpu`: moves all tensors to cpu
    :func:`cuda`: moves all tensors to gpu
    :func:`cuda_or_cpu`: matches optimizer and scheduler states to model device
    :func:`fit`: performs model training

    And redefines the following methods:

    :func:`__init__`: instantiates CNNClassifier objects
    :func:`build`: assembles a convolutional neural network
    """

    def __init__(
        self,
        activation=torch.nn.ReLU,
        batch_size=128,
        conv_layers=(5, 5),
        dropout=0.0,
        hidden_layers=(120, 84),
        iters=10,
        kernel_size=3,
        learning_rate=1e-3,
        loss=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        safe_batch=True,
        scheduler=None,
        scheduler_params={},
        threads=None,
        verbosity=0.25,
    ):
        """
        This method instantiates CNNClassifier objects with a variety of
        attributes necessary to support model training and logging.
        Importantly, models are not usable until they are trained (via fit()),
        this "lazy" model creation schema allows us to abstract parameterizing
        of number of features and labels (thematically similar to scikit-learn
        model classes) on initialization.

        :param activation: activation function at each hidden layer
        :type activation: torch nn class
        :param batch_size: training batch size (-1 for 1 batch)
        :type batch_size: int
        :param conv_layers: the number of filters at each convolutional layer
        :type conv_layers: tuple of ints
        :param dropout: dropout probability
        :type dropout: float
        :param hidden_layers: the number of neurons at each hidden layer
        :type hidden_layers: tuple of ints
        :param iters: number of training iterations (ie epochs)
        :type iters: int
        :param kernel_size: size of the convolving kernel
        :type kernel_size: int
        :param learning_rate: learning rate
        :type learning_rate: float
        :param loss: loss function
        :type loss: torch nn class
        :param optimizer: optimizer from optimizers package
        :type optimizer: torch optim class
        :param optimizer_params: optimizer parameters
        :type optimizer_params: dict
        :param safe_batch: prevent gpu oom errors for large batches
        :type safe_batch: bool
        :param scheduler: scheduler for dynamic learning rates
        :type scheduler: torch optim.lr_scheduler class or None
        :param scheduler_params: scheduler parameters
        :type scheduler_params: dict
        :param threads: number of cpu threads used for training
        :type threads: int or None (for max threads)
        :param verbosity: print training statistics every verbosity%
        :type verbosity: float
        :return: a convolutional neural network classifier skeleton
        :rtype: CNNClassifier object
        """
        super().__init__(
            activation,
            batch_size,
            dropout,
            hidden_layers,
            iters,
            learning_rate,
            loss,
            optimizer,
            optimizer_params,
            safe_batch,
            scheduler,
            scheduler_params,
            threads,
            verbosity,
        )
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.params.update({"conv_layers": conv_layers, "kernel_size": kernel_size})
        return None

    def build(self, x, y, shape):
        """
        Like the implementation of build() from the MLPClassifier class, this
        method instantiates a torch Sequential object. It additionally adds
        support for interleaving activation functions, convolutional layers,
        and maxpool layers. This abstraction allows us to dynamically build
        models based on the passed-in dataset, versus hardcoding model
        architectures via a forward() method. Critically, this method asssumes
        that inputs are flattened vectors to be reshaped according to shape for
        interoperability with https://github.com/sheatsley/datasets.

        :param x: dataset of inputs
        :type x: torch Tensor object (n, m)
        :param y: dataset of labels
        :type y: Pytorch Tensor object (n,)
        :param shape: original shape of x (channels, width, height)
        :type shape: tuple of ints
        :return: an untrained convolutional neural network classifier
        :rtype: torch Sequential object
        """

        # assemble convolutional and linear layers
        self.params["shape"] = shape
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
        linear = super().build(x, y)

        # attach unflatten and meld convolutional and linear layers with flatten
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, shape), *convolutional, torch.nn.Flatten(), *linear
        )


if __name__ == "__main__":
    """ """
    raise SystemExit(0)

# Deep Learning Models

_Deep Learning Models_ (`dlm`) is a repo for building supervised,
PyTorch-based, deep learning models with simple,
[Keras](https://keras.io)-like, interfaces. Designed with [adversarial machine
learning](https://arxiv.org/abs/1412.6572.pdf) in mind, `dlm` makes it easy to
train and evaluate the performance of models on clean inputs and adversarial
examples.  This repo is, by design, to be interoperable with the following
[datasets](https://github.com/sheatsley/datasets) and
[attacks](https://github.com/sheatsley/datasets) (for adversarial training)
repos (which are all based on [PyTorch](https://github.com/pytorch/pytorch)).
Some features of this repo include [optimal templates for
datasets](https://github.com/sheatsley/models/blob/main/dlm/templates.py),
[automatic batch sizing to prevent OOM errors on
GPUs](https://github.com/sheatsley/models/blob/main/dlm/models.py#L366),
implicit support for adversarial training subroutines, among others. All of the
information you need to start using this repo is contained within this one
ReadMe, ordered by complexity (No need to parse through any ReadTheDocs
documentation).

## Table of Contents

* [Quick Start](#quick-start)
* [Advanced Usage](#advanced-usage)
* [Repo Overview](#repo-overview)
* [Parameters](#parameters)

## Quick Start

While I recommend using the [datasets](https://github.com/sheatsley/datasets)
repo, it should be trivial to bring your own data. The only notable difference
between this repo and other deep learning libraries is that it assumes all
inputs are flattened, including images; `CNNClassifier` objects accept a
`shape` argument on initialization that is used with `torch.nn.Unflatten` to
ensure tensors have the correct shape prior to applying convolutions. Using the
[attacks](https://github.com/sheatsley/attacks) repo makes it easy to perform
adversarial training with custom attacks (versus bringing your own algorithms).
Finally, I recommend installing an editable version of this repo via `pip
install -e`. Afterwards, you can train a simple CNN on
[MNIST](http://yann.lecun.com/exdb/mnist/) with
[PGD](https://arxiv.org/pdf/1706.06083.pdf)-based adversarial training as
follows:

```
import aml
import dlm
import mlds
import torch

# load data
mnist = mlds.mnist
x_train = torch.from_numpy(mnist.train.data)
y_train = torch.from_numpy(mnist.train.labels).long()
x_test = torch.from_numpy(mnist.test.data)
y_test = torch.from_numpy(mnist.test.labels).long()

# instantiate and adversarially train a model
model = dlm.CNNClassifier(activation=torch.nn.ReLU,
            attack=aml.pgd, 
            attack_params=dict(
               alpha=0.01,
               epochs=40,
               epsilon=0.3,
            ),
            conv_layers=(32, 64),
            dropout=0.0,
            hidden_layers=(1024,),
            kernel_size=1,
            learning_rate=1e-4,
            loss=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer_params={}
            scheduler=None,
            scheduler_params={},
            shape=(1, 28, 28),
)
model.fit(x_train, y_train)

# plot the model loss over the training epochs
model.res.plot(x="epoch", y="training_loss")
```

Other uses can be found in the
[examples](https://github.com/sheatsley/models/tree/main/examples) directory.

## Advanced Usage

Below are descriptions of some of the more subtle controls within this repo and
complex use cases.

* Adversarial training: When an `aml.Attack` (or `Adversary`) object is
    provided on model initialization, then models are [adversarially
    trained](https://arxiv.org/pdf/1412.6572.pdf). At this time, three
    adversarial training schemes are supported: [à la
    Mądry](https://arxiv.org/pdf/1706.06083.pdf) (sometimes called "PGD-AT"),
    [TRADES](https://arxiv.org/pdf/1901.08573.pdf), and
    "[Free](https://arxiv.org/pdf/1904.12843.pdf)." Implementation details
    surrounding these regimes can be found in `dlm.training`

* Auto-batch: `RuntimeError: CUDA out of memory.` errors are certainly in the
    running for the most common runtime error with deep learning backends like
    PyTorch (perhaps only second to `RuntimeError: shape mismatch`, but at
    least [lazy
    modules](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html)
    are here to save us from that). _Auto-batching_ is a feature within all
    `dlm.models` classes that attempts to find the maximal batch size for three
    use cases: training, inference, and crafting adversarial examples. Upon
    model initialization, if `auto_batch` is `True`, then, on `fit`, the
    `max_batch_size` subroutine is called. This subroutine performs binary
    search (by catching exceptions) over these three use cases with randomly
    generated data of varying batch sizes. Once the batch sizes are determined,
    the model is trained normally, and any interactions with the model are
    subsequently batched to the appropriate size (based on whether the model,
    the inputs, or neither, are tracking gradients).

* Classes: When `classes` is not provided on model initialization, it will be
    inferred from the number of unique elements on the label set during
    training. In this scenario, `y` when passed into `fit` should contain
    representatives from each class so that the number of outputs can be
    computed correctly.

* Threads: When training on the CPU, `threads` determines the number of
    available threads to use for training. I recommend investigating the
    parameter with some toy problems; empirically, using 50% of the available
    threads appears to yield the fastest training time for typical batch sizes.

## Repo Overview

This repo was designed for interoperability with following
[datasets](https://github.com/sheatsley/datasets) and
[attacks](https://github.com/sheatsley/datasets) repos. The intention is to
enable rapid prototyping for evaluating the robustness of deep learning models
to adversarial examples. To this end, the abstractions defined in this repo are
heavily inspired by [Keras](https://keras.io) and
[scikit-learn](https://scikit-learn.org/stable/): simple, but (reasonably)
limited. Within `dlm.models` (the core of `dlm`) there are (currently) three
classes: `LinearClassifier`, `MLPClassifier` (which inherits from
`LinearClassifier`), and `CNNClassifier` (which inherits from `MLPClassifier`).
The `dlm.templates` module provides hyperparameters I have found that are
competitive with the state-of-the-art, prioritizing a lower training time
(e.g., lower epochs, larger batch sizes, and smaller models). Finally, the
`dlm.training` module provides varying flavors of [adversarial
training](https://arxiv.org/pdf/1412.6572.pdf).

Beyond specific parameters (described below), the following are some useful
methods: `accuracy`: returns model accuracy over some data points, `fit`:
trains the model on a dataset, `load`: loads pretrained models (either
[sequential
containers](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
or
[state_dicts](https://pytorch.org/tutorials/beginner/saving_loading_models.html)),
`save`: saves models, and `to`: moves the model, optimizer, and scheduler to a
device.

## Parameters

Most initialization parameters are self-explanatory, but are listed here for
reference.


### Linear Classifier

`LinearClassifer` objects compile the simplest models and principally serves to
define all of the bells and whistles that make training, evaluating, and
manipulating models easy. They accept the following parameters:

* `attack`: attack to use when performing adversarial training
* `attack_params`: parameters to configure the attack
* `auto_batch`: automatically determine the max batch size when using GPUs
* `batch_size`: training batch size (-1 for 1 batch)
* `classes`: number of classes
* `device`: hardware device to use
* `epochs`: number of training iterations
* `learning_rate`: determines the pace of parameter updates
* `loss`: loss function to use
* `optimizer`: optimizer to use
* `optimizer_params`: parameters to configure the optimizer
* `scheduler`: scheduler to use
* `scheduler_params`: parameters to configure the scheduler
* `threads`: number of threads to use when using the CPU
* `verbosity`: print training statistics every verbosity%

### Multilayer Perceptron

`MLPClassifer` objects extend `LinearClassifier` to include support for
activation functions, dropout, and hidden layers. These models are compiled
with a set of "component blocks", where the number of blocks is determined by
the number of hidden layers. Each block is defined as a dropout layer, a fully
connected layer, and an activation function. Then, a final fully connect layer
is appended on the end (via `LinearClassifier` parent class), which returns the
model logits. In addition to `LinearClassifier` arguments, the following
parameters are also accepted:

* `activation`: activation function to use
* `dropout`: dropout probability 
* `hiddden_layers`: the number of neurons at each hidden layer

### Convolutional Neural Networks

`CNNClassifier` objects extend `MLPCLassifier` to include support for
convolutional layers with parameterized kernel sizes. Like `MLPClassifier`
objects, models are compiled with a set of "component blocks", where the number
of (convolutional) blocks is determined by the number of convolutional layers.
Each block is defined as a convolutional layer, an activation function, and a
max pool layer. Afterwards, a flatten layer is added and the fully connected
portion of the model is compiled via the `MLPClassifier` parent class. Notably,
an
[unflatten](https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html)
layer is prepended to the beginning of the model, as data processed via the
[datasets](https://github.com/sheatsley/datasets) repo is always flattened
(which I find makes working with adversarial examples a simpler process). In
addition to `MLPClassifer` arguments, the following parameters are also
accepted:

* `conv_layers`: number of filters at each convolutional layer
* `kernel_size`: size of the convolving kernel
* `shape`: original input shape (used to unflatten inputs)

# Deep Learning Models

_Deep Learning Models_ (`dlm`) is a repo for building PyTorch-based deep
learning models with simple, [Keras](https://keras.io)-like, interfaces. Designed with [adversarial
machine learning](https://arxiv.org/abs/1412.6572.pdf) in mind, `dlm` makes it
easy to train and evaluate the performance of models on clean inputs and
adversarial examples.  This repo is, by design, to be interoperable with the
following [datasets](https://github.com/sheatsley/datasets) and
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
* [Advanced](#quick-start)
* [Repo Overview](#repo-overview)
* [Citation](#citation)

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
hyperparameters = dlm.hyperparameters.mnist
model = dlm.CNNClassifier(**hyperparameters)
model.fit(x_train, y_train)

# set attack parameters and produce adversarial perturbations
step_size = 0.01
number_of_steps = 30
budget = 0.15
pgd = aml.pgd(step_size, number_of_steps, budget, model)
perturbations = pgd.craft(x_test, y_test)

# compute some interesting statistics and publish a paper
accuracy = model.accuracy(x_test + perturbations, y_test)
mean_budget = perturbations.norm(torch.inf, 1).mean()
```

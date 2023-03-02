# Deep Learning Models

_Deep Learning Models_ (`dlm`) is a repo for building PyTorch-based deep
learning models with simple, Keras-like, interfaces. Designed with [adversarial
machine learning](https://arxiv.org/abs/1412.6572) in mind, `dlm` makes it easy
to train and evaluate the performance of models on clean inputs and adversarial
examples. Its design is intended for interoperability with the following
[datasets](https://github.com/sheatsley/datasets) and
[attacks](https://github.com/sheatsley/attacks) libraries (which are all based
on [PyTorch](https://github.com/pytorch/pytorch)). Some features of this repo
include [optimal templates for
datasets](https://github.com/sheatsley/models/blob/main/dlm/templates.py),
[automatic batch sizing to prevent OOM errors on
GPUs](https://github.com/sheatsley/models/blob/main/dlm/models.py#L366),
implicit support for adversarial training subroutines, among others. All of the
information you need to start using this repo is contained within this one
ReadMe, ordered by complexity (No need to parse through any ReadTheDocs
documentation).

## Table of Contents

* [Quick start](#quick-start)
* [Advanced](#quick-start)
* [Repo Overview](#repo-overview)
* [Citation](#citation)

# Deep Learning Models Examples

This directory contains various examples showing how `dlm` can be used. Note
that some of the examples will write figures to this directory.

* `best_batch_size.py`: performs model training over a variety of batch sizes
and plots model accuracy over batch size on training and validation data.
* `cpu_vs_metal_training.py`: measures and plots the wall-clock time of training
models, performing inference, and crafting adversarial examples. (Requires
[mlds](https://github.com/sheatsley/datasets) and
[aml](https://github.com/sheatsley/attacks))
* `hyperparameter_tuning.py`: performs (crude) hyperparameter tuning and
produces a parallel coordinates plot, colored by the validation loss.

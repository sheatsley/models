# Deep Learning Models Examples

This directory contains various examples showing how `dlm` can be used. Note
that these examples will write figures to this directory (and depend on
`plotly`, `matplotlib`, or `seaborn`). All of these examples depend on the
[mlds](https://github.com/sheatsley/datasets) repo.

* `best_batch_size.py`: performs model training over a variety of batch sizes
*   and plots model accuracy over batch size on training and validation data.
* `cpu_vs_metal_training.py`: measures and plots the wall-clock time of training
    models, performing inference, and crafting adversarial examples. This
    example also requires the [aml](https://github.com/sheatsley/attacks) repo.
* `hyperparameter_tuning.py`: performs (basic) hyperparameter tuning and
    produces a parallel coordinates plot, colored by the validation loss.

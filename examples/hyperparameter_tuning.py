"""
This script performs basic hyperparameter tuning and produces a parallel
coordinates plot over the parameter space.
"""
import argparse
import itertools
import warnings

import dlm
import mlds
import pandas
import plotly.express as px
import torch

# dlm uses lazy modules which induce warnings that overload stdout
warnings.filterwarnings("ignore", category=UserWarning)


def main(batch_sizes, dataset, device, epochs, hidden_layers, learning_rates):
    """
    This function is the main entry point for hyperparameter tuning.
    Specifically, this: (1) loads the dataset, (2) computes the parameter space
    combinatorically, (3) trains models, and (4) plots the results.

    :param batch_sizes: batch sizes to consider
    :type batch_sizes: tuple of ints
    :param dataset: dataset to use
    :type dataset: str
    :param device: hardware device to train models on
    :type device: str
    :param epochs: epochs to consider
    :type epochs: tuple of ints
    :param hidden_layers: hidden_layers to consider
    :type hidden_layers: tuple of ints
    :param learning_rates: learning rates to consider
    :type learning_rates: tuple of floats
    :return: None
    :rtype: NoneType
    """

    # load dataset, architecture, and base parameters
    data = getattr(mlds, dataset)
    try:
        x = torch.from_numpy(data.train.data)
        y = torch.from_numpy(data.train.labels).long()
        xv = torch.from_numpy(data.test.data)
        yv = torch.from_numpy(data.test.labels).long()
        has_test = True
    except AttributeError:
        has_test = False
        x = torch.from_numpy(data.dataset.data)
        y = torch.from_numpy(data.dataset.labels).long()
    template = getattr(dlm.templates, dataset)
    architecture, hyperparameters = (
        (dlm.CNNClassifier, template.cnn)
        if hasattr(template, "cnn")
        else (dlm.MLPClassifier, template.mlp)
    )

    # set hidden layer neurons to be spaced between features and classes
    f, c = x.size(1), template.classes
    hidden = tuple(
        tuple(int((f - c) / (h + 1) * n + c) for n in range(h, 0, -1))
        for h in hidden_layers
    )

    # make prints nice, compute parameter space, and set results dataframe
    ba, ea, ha, la = (
        len(str(max(p, key=lambda p: len(str(p)))))
        for p in (batch_sizes, epochs, hidden, learning_rates)
    )
    space = tuple(itertools.product(batch_sizes, epochs, hidden, learning_rates))
    metrics = (
        "batch_size",
        "epoch",
        "hidden_layers",
        "learning_rate",
        "validation_accuracy",
    )
    results = pandas.DataFrame(0, index=range(len(space)), columns=metrics)
    print(f"Total parameter space to consider: {len(space)}")

    # instantiate models, override parameters, and perform training
    for i, (b, e, h, l) in enumerate(space):
        model = architecture(
            **hyperparameters
            | dict(
                batch_size=b,
                device=device,
                epochs=e,
                hidden_layers=h,
                learning_rate=l,
                verbosity=0,
            )
        )
        x, y = x.to(device), y.to(device)
        model.fit(x, y, valset=(xv, yv) if has_test else 0.2)
        results.loc[i] = (b, e, str(h), l, model.res.validation_accuracy.iloc[-1])
        print(
            f"Completed bsize={b:{ba}}, epochs={e:{ea}}, hlayers={str(h):>{ha}},",
            f"lrate={l:{la}}, T Acc: {model.res.training_accuracy.iloc[-1]:6.2%}",
            f"V Acc: {model.res.validation_accuracy.iloc[-1]:6.2%}",
            f"Hparam {i} of {len(space)} ({i/len(space):6.2%})",
        )

    # plot and save the results
    plot(dataset, results)
    return None


def plot(dataset, results):
    """
    This function plots hyperparameter tuning results. Specifically, this
    produces a parallel coordiantes plot over the parameter space. Notably,
    this plot assumes that all entries in the dataframe are floats in 0-1. The
    plot is written to disk in the current directory.

    :param dataset: dataset used
    :type dataset: str
    :param results: results of the performance measurements
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    fig = px.parallel_categories(
        data_frame=results,
        color="validation_accuracy",
        color_continuous_scale=px.colors.diverging.BrBG,
        range_color=(0, 1),
        title=f"{dataset}",
    )
    fig.write_image(__file__[:-3] + f"_{dataset}.pdf")
    return None


if __name__ == "__main__":
    """
    This script performs basic hyperparameter tuning (i.e., a grid search).
    Specifically, this script: (1) combinatorically iterates over the provided
    parameters, (2) trains models over current parameter set (concurrently, if
    possible), (3) measures validation accuracy, and (4) produces a parallel
    coordinates plot. Datasets are provided by mlds
    (https://github.com/sheatsley/datasets).
    """
    parser = argparse.ArgumentParser(
        conflict_handler="resolve", description="Basic hyperparameter tuning"
    )
    parser.add_argument(
        "-b",
        "--bsizes",
        default=(32, 64, 128),
        help="Batch sizes to consider",
        nargs="+",
        type=lambda b: sorted(map(int, b)),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=mlds.__available__,
        default="phishing",
        help="Dataset to use",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Hardware device to train models on",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=(5, 10, 20, 40),
        help="Training epochs to consider",
        nargs="+",
        type=lambda b: sorted(map(int, b)),
    )
    parser.add_argument(
        "-h",
        "--hlayers",
        default=(1, 2),
        help="Number of hidden layers (neurons computed automatically)",
        nargs="+",
        type=lambda h: sorted(map(int, h)),
    )
    parser.add_argument(
        "-l",
        "--lrates",
        default=(10**-x for x in range(2, 5)),
        help="Learning rates to consider",
        nargs="+",
        type=lambda r: sorted(map(float, r)),
    )
    args = parser.parse_args()
    main(
        batch_sizes=tuple(args.bsizes),
        dataset=args.dataset,
        device=args.device,
        epochs=tuple(args.epochs),
        hidden_layers=tuple(args.hlayers),
        learning_rates=tuple(args.lrates),
    )
    raise SystemExit(0)

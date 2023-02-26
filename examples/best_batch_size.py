"""
This script measures model performance as a function of batch size.
Author: Ryan Sheatsley
Fri Feb 24 2023
"""
import dlm  # PyTorch-based deep learning models with Keras-like interfaces
import argparse  # Parser for command-line options, arguments and sub-commands
import mlds  # Scripts for downloading, preprocessing, and numpy-ifying popular machine learning datasets
import pandas  # Powerful data structures for data analysis, time series, and statistics
import seaborn  # statistical data visualization
import torch  # Tensors and Dynamic neural networks in Python with strong GPU acceleration
import warnings  # Warning control
import matplotlib.pyplot as plt

# dlm uses lazy modules which induce warnings that overload stdout
warnings.filterwarnings("ignore", category=UserWarning)


def plot(results):
    """
    This function plots the batch size performance measurement results.
    Specifically, this produces one line plot per dataset containing model loss
    on training & validation data over (normalized) epochs. Batch sizes are
    divided by color, while data and validation accuracies are split by line
    style. The plot is written to disk in the current directory.

    :param results: results of the performance measurements
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    plot = seaborn.relplot(
        data=results,
        col="dataset",
        col_wrap=(results.dataset.unique().size + 1) // 2,
        kind="line",
        hue="batch size",
        palette="flare",
        style="data",
        x="epoch",
        y="loss",
    )
    plot.savefig(
        "/".join(__file__.split("/")[:-1]) + "/best_batch_size",
        bbox_inches="tight",
    )
    return None


def main(batch_sizes, datasets, device):
    """
    This function is the main entry point for the batch size benchmark.
    Specifically, this: (1) loads the dataset(s), (2) iterates over batch
    sizes, (3) trains models & reports loss, and (4) plots the results.

    :param batch_sizes: batch sizes to test
    :type batch_sizes: tuple of ints
    :param dataset: dataset to use
    :type dataset: str
    :param device: hardware device to train models on
    :type device: str
    :return: None
    :rtype: NoneType
    """
    results = pandas.DataFrame(
        columns=("batch size", "dataset", "data", "epoch", "loss"),
    )
    for i, dataset in enumerate(datasets):

        # load dataset and convert to pytorch tensors
        print(f"Preparing {dataset} dataset... {i + 1} of {len(datasets)}")
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

        # fetch model template
        template = getattr(dlm.templates, dataset)
        architecture, hyperparameters = (
            (dlm.CNNClassifier, template.cnn)
            if hasattr(template, "cnn")
            else (dlm.MLPClassifier, template.mlp)
        )

        # instantiate models, iterate over batch sizes, and perform training
        for j, b in enumerate(batch_sizes):
            print(f"On {dataset} batch size {b}... {j} of {len(batch_sizes)} ")
            model = architecture(
                **hyperparameters | dict(batch_size=b, device=device, verbosity=0)
            )
            x, y = x.to(device), y.to(device)
            model.fit(x, y, valset=(xv, yv) if has_test else 0.2)

            # prepare the results and extend the dataframe
            model.res.training_loss /= model.res.training_loss.max()
            model.res.validation_loss /= model.res.validation_loss.max()
            result = model.res.melt(
                id_vars=["epoch"],
                value_name="loss",
                value_vars=["training_loss", "validation_loss"],
                var_name="data",
            ).replace("_loss", "", regex=True)
            result.epoch /= result.epoch.max()
            result["batch size"] = b
            result["dataset"] = dataset
            results = pandas.concat((results, result))

    # plot and save the results
    plot(results)
    return None


if __name__ == "__main__":
    """
    This script measures model performance as a function of batch size.
    Specifically, this script: (1) trains models over a set of batch sizes, and
    (2) measures and plots model loss on training and validation data over the
    (normalized) number of epochs. Datasets are provided by mlds
    (https://github.com/sheatsley/datasets),
    """
    parser = argparse.ArgumentParser(description="batch size performance benchmarks")
    parser.add_argument(
        "-b",
        "--bsizes",
        default=(min(2**i, 256 * max(1, i - 7)) for i in range(5, 15)),
        help="Batch sizes to benchmark over",
        nargs="+",
        type=lambda b: map(int, b),
    )
    parser.add_argument(
        "-d",
        "--datasets",
        choices=mlds.__available__,
        default=mlds.__available__,
        help="Dataset(s) to benchmark performance with",
        nargs="+",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Hardware device to train models on",
    )
    parser.add_argument(
        "-e",
        "--equalize",
        action="store_true",
        help="Equalize the total number of backprops (experimental)",
    )
    args = parser.parse_args()
    main(batch_sizes=tuple(args.bsizes), datasets=args.datasets, device=args.device)
    raise SystemExit(0)

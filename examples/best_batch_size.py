"""
This script measures model performance as a function of batch size.
Author: Ryan Sheatsley
Fri Feb 24 2023
"""
import argparse
import warnings

import dlm
import mlds
import pandas
import seaborn
import torch

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
        facet_kws=dict(sharex=False),
        hue="batch size",
        kind="line",
        legend="full" if results.dataset.unique().size > 1 else "auto",
        palette="flare",
        style="data",
        x="epoch",
        y="loss",
    )
    plot.savefig(__file__[:-3] + ".pdf")
    return None


def main(batch_sizes, datasets, device, equalize):
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
    :param equalize: ensures all models have the same number of backprops
    :param equalize: bool
    :return: None
    :rtype: NoneType
    """
    results = pandas.DataFrame(
        columns=("batch size", "dataset", "data", "epoch", "loss"),
    )
    print(f"Batch sizes to benchmark: {batch_sizes}")
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

        # fetch model template and equalize backprops if necessary
        template = getattr(dlm.templates, dataset)
        architecture, hyperparameters = (
            (dlm.CNNClassifier, template.cnn)
            if hasattr(template, "cnn")
            else (dlm.MLPClassifier, template.mlp)
        )
        backprops = hyperparameters["epochs"] / batch_sizes[0]

        # instantiate models, iterate over batch sizes, and perform training
        for j, b in enumerate(batch_sizes):
            print(f"On {dataset} batch size {b}... {j + 1} of {len(batch_sizes)} ")
            epochs = int(backprops * b) if equalize else hyperparameters["epochs"]
            model = architecture(
                **hyperparameters
                | dict(batch_size=b, device=device, epochs=epochs, verbosity=0)
            )
            x, y = x.to(device), y.to(device)
            model.fit(x, y, valset=(xv, yv) if has_test else 0.2)

            # prepare the results and extend the dataframe
            result = model.res.melt(
                id_vars=["epoch"],
                value_name="loss",
                value_vars=["training_loss", "validation_loss"],
                var_name="data",
            ).replace("_loss", "", regex=True)
            result["batch size"] = b
            result["dataset"] = dataset
            results = pandas.concat((results, result))

        # normalize loss results
        for data in ("validation", "training"):
            mask = (results.dataset == dataset) & (results.data == data)
            results.loc[mask, "loss"] /= results.loc[mask, "loss"].max()

    # normalize results, plot them, and write to disk
    plot(results)
    return None


if __name__ == "__main__":
    """
    This script measures model performance as a function of batch size.
    Specifically, this script: (1) trains models over a set of batch sizes, and
    (2) measures and plots model loss on training and validation data over the
    (normalized) number of epochs. Datasets are provided by mlds
    (https://github.com/sheatsley/datasets).
    """
    parser = argparse.ArgumentParser(description="batch size performance benchmarks")
    parser.add_argument(
        "-b",
        "--bsizes",
        default=(min(2**i, 256 * max(1, i - 7)) for i in range(4, 15)),
        help="Batch sizes to benchmark over",
        nargs="+",
        type=lambda b: sorted(map(int, b)),
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
    main(
        batch_sizes=tuple(args.bsizes),
        datasets=args.datasets,
        device=args.device,
        equalize=args.equalize,
    )
    raise SystemExit(0)

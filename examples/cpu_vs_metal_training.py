"""
This script benchmarks (and produces a bar chart of) the performance of
training, inference, and crafting adversarial examples on macOS cpus vs gpus.
"""
import argparse
import os
import platform
import time

import aml
import dlm
import matplotlib.pyplot as plt
import mlds
import pandas
import seaborn
import torch

# necessary until https://github.com/pytorch/pytorch/issues/77764 is resolved
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def benchmark(attack, model, x, y):
    """
    This function benchmarks model training, inference, and crafting
    adversarial examples. Specifically, each benchmark measures wall-clock time
    for the following calls: (1) model.fit (training), (2) model (inference),
    and (3) attack.craft (crafting). This is designed to provide a rough
    estimate of the performance in realistic workloads; there are better
    metrics than wall-clock time for more accurate performance measurements.

    :param attack: attack to craft adversarial examples with
    :type attack: aml Attack object
    :param model: model to use
    :type model: dlm LinearClassifier-inherited object
    :param trials: number of trials to perform for each benchmark
    :type trials: int
    :param x: data
    :type x: torch Tensor object (n, m)
    :param y: labels
    :type y: torch Tensor object (n,)
    :return: wall-clock times for training, inference, and crafting
    :rtype: tuple of floats
    """
    times = []
    stages = "training", "inference", "crafting"
    for stage, benchmark in zip(stages, (model.fit, model, attack.craft)):
        print(f"Performing {stage}...", end="\r")
        start = time.time()
        benchmark(x) if benchmark is model else benchmark(x, y)
        times.append(time.time() - start)
        print(f"{stage} complete! ({times[-1]:.2f}s)")
    return times


def main(attack, datasets):
    """
    This function is the main entry point for macOS cpu vs gpu comparison.
    Specifically, this: (1) loads the dataset(s), (2) instantiates the attack
    object, (3) benchmarks performance, and (4) plots the results.

    :param attack: attack to use
    :type attack: aml function
    :param dataset: dataset to use
    :type dataset: str
    :return: None
    :rtype: NoneType
    """
    results = pandas.DataFrame(
        0,
        index=range(2 * len(datasets)),
        columns=("device", "dataset", "training", "inference", "crafting"),
    )

    # load dataset and convert to pytorch tensors
    for i, dataset in enumerate(datasets):
        print(f"Preparing {dataset} dataset... {i + 1} of {len(datasets)}")
        data = getattr(mlds, dataset)
        try:
            x = torch.from_numpy(data.train.data)
            y = torch.from_numpy(data.train.labels).long()
        except AttributeError:
            x = torch.from_numpy(data.dataset.data)
            y = torch.from_numpy(data.dataset.labels).long()

        # compute feature range for l2 attacks and fetch model template
        mins, maxs = (x.min(0).values.clamp(max=0), x.max(0).values.clamp(min=1))
        template = getattr(dlm.templates, dataset)
        architecture, hyperparameters = (
            (dlm.CNNClassifier, template.cnn)
            if hasattr(template, "cnn")
            else (dlm.MLPClassifier, template.mlp)
        )

        # instantiate an mlp or cnn and migrate tensors to the device
        for j, d in enumerate(("cpu", "mps")):
            print(f"Instantiating {architecture.__name__} model on {d}...")
            model = architecture(**hyperparameters | dict(device=d, verbosity=0))
            x, y = x.to(d), y.to(d)

            # instantiate the attack and perform the benchmarks
            print(f"Instantiating {attack.__name__} on {d} and running benchmarks...")
            attack_parameters = dict(alpha=0.01, epochs=15, model=model, verbosity=0)
            budget = 0.15
            l0 = int(x.size(1) * budget) + 1
            l2 = maxs.sub(mins).norm(2).mul(budget).item()
            linf = budget
            budgets = {
                **dict.fromkeys((aml.apgdce, aml.apgddlr, aml.bim, aml.pgd), linf),
                **dict.fromkeys((aml.cwl2, aml.df, aml.fab), l2),
                **dict.fromkeys((aml.jsma,), l0),
            }
            atk = attack(**attack_parameters | dict(epsilon=budgets[attack]))
            results.loc[i * 2 + j] = [d, dataset] + benchmark(atk, model, x, y)

    # plot and save the results
    plot(results)
    return None


def plot(results):
    """
    This function plots the performance measurement results. Specifically, this
    produces a series of a clustered, stacked, horizontal bar charts of the
    measured wall-clock times (in seconds) corresponding to training,
    inference, and crafting times. A dataset-number of clusters are produced,
    with each cluster containing two bars (i.e., cpu and gpu times). The plot
    is written to disk in the current directory.

    :param results: results of the performance measurements
    :type results: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    title = (
        f"{platform.platform()}, torch v{torch.__version__}, aml {aml.__version__}, "
        f"dlm {dlm.__version__}, {time.strftime('%m/%d/%y', time.localtime())}"
    )
    results = results.sort_values("training", ascending=False)
    stages = "training", "crafting", "inference"
    seaborn.set_theme(style="whitegrid")
    _, ax = plt.subplots()
    for stage, palette in zip(stages, ("pastel", "muted", "dark")):
        seaborn.barplot(
            data=results,
            hue="device",
            palette=palette,
            x=stage,
            y="dataset",
        )
    seaborn.despine(left=True, bottom=True)
    ax.set(title=title, xlabel="seconds", xscale="log", ylabel="")
    elements = zip(*ax.get_legend_handles_labels())
    handles, labels = zip(
        *((h, s) for (h, _), s in zip(sorted(elements, key=lambda x: x[1]), stages * 2))
    )
    mps_legend = ax.legend(
        bbox_to_anchor=(1, 1),
        frameon=False,
        handles=handles[len(stages) :],
        labels=labels[len(stages) :],
        title="mps",
    )
    ax.add_artist(mps_legend)
    ax.legend(
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        handles=handles[: len(stages)],
        labels=labels[: len(stages)],
        title="cpu",
    )
    ax.get_figure().savefig(__file__[:-3] + ".pdf", bbox_inches="tight")
    return None


if __name__ == "__main__":
    """
    This script benchmarks the performance of training, inference, and crafting
    adversarial examples on macOS cpus vs gpus (i.e., Metal). Datasets are
    provided by mlds (https://github.com/sheatsley/datasets) and attacks by
    (https://github.com/sheatsley/attacks). Specifically, this script: (1)
    parses command-line arguments, (2) loads dataset(s), (3) collects runtime
    statistics on model training, test set inference, and crafting adversarial
    examples on the both the cpu and gpu, and (4) plots the results.
    """
    parser = argparse.ArgumentParser(
        description="macOS cpu vs gpu performance comparison"
    )
    parser.add_argument(
        "-a",
        "--attack",
        choices=(a for a in dir(aml) if callable(getattr(aml, a))),
        default="bim",
        help="Attack to craft adversarial examples with",
        type=lambda x: getattr(aml, x),
    )
    parser.add_argument(
        "-d",
        "--datasets",
        choices=mlds.__available__,
        default=mlds.__available__,
        help="Dataset(s) to benchmark performance with",
        nargs="+",
    )
    args = parser.parse_args()
    main(attack=args.attack, datasets=args.datasets)
    raise SystemExit(0)

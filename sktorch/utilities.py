"""
The utilities module defines functions used throughout Scikit-Torch.
Author: Ryan Sheatsley
Tue Dec 6 2022
"""
import builtins  # Built-in objects
import contextlib  # Utilities for with-statement contexts
import time  # Time access and conversions


def print(*args, **kwargs):
    """
    This function wraps the print function, prepended with a timestamp.

    :param *args: positional arguments supported by print()
    :type *args: tuple
    :param **kwargs: keyword arguments supported by print()
    :type **kwargs: dictionary
    :return: None
    :rtype: NoneType
    """
    return builtins.print(f"[{time.asctime()}]", *args, **kwargs)


@contextlib.contextmanager
def supress_stdout(enabled):
    """
    This function conditionally supresses writes to stdout. This is
    particularly useful in synchronizing the verbosity argument to models with
    libraries that may arbitrarily print information.

    :param enabled: whether writes to stdout are supressed
    :type enabled: bool
    :return: None
    :rtype: iterator
    """
    with contextlib.redirect_stdout(None) if enabled else contextlib.nullcontext():
        yield None


if __name__ == "__main__":
    """
    This runs some basic unit tests with the functions defined in this module
    """
    print("Test string with implicit date.")
    raise SystemExit(0)

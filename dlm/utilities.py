"""
The utilities module defines functions used throughout dlm.
Author: Ryan Sheatsley
Tue Dec 6 2022
"""
import contextlib  # Utilities for with-statement contexts


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

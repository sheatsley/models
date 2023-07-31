"""
Build script for Deep Learning Models
"""
import subprocess

import setuptools
import setuptools.command.install


class Install(setuptools.command.install.install):
    """
    This class overrides the default install command so that the __version__
    attribute of the dlm package is set statically.

    :func:`run`: computes the git hash and saves it to a file
    """

    def run(self):
        """
        This method computes the git hash and saves it to a file.

        :return: None
        :rtype: Nonetype
        """
        version = subprocess.check_output(
            ("git", "rev-parse", "--short", "HEAD"), text=True
        ).strip()
        with open("mlds/VERSION", "w") as f:
            f.write(f"{version}\n")
        return super().run()


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    cmdclass=dict(install=Install),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="A PyTorch-based Keras-inspired deep learning library",
    install_requires=["pandas", "torch"],
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine-learning pytorch keras",
    name="dlm",
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    url="https://github.com/sheatsley/models",
    version="1.1",
)

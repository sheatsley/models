"""
Build script for Deep Learning Models
"""
import subprocess

import setuptools

# compute git hash and save to file for non-editable installs
# overriding install for package data is bugged: https://github.com/pypa/setuptools/issues/1064
version = subprocess.check_output(
    ("git", "rev-parse", "--short", "HEAD"), text=True
).strip()
with open("dlm/VERSION", "w") as f:
    f.write(f"{version}\n")


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
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
    package_data={"dlm": ["VERSION"]},
    python_requires=">=3.10",
    url="https://github.com/sheatsley/models",
    version="1.1",
)

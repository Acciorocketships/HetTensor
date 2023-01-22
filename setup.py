from setuptools import setup
from setuptools import find_packages

setup(
    name="hettensor",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A wrapper for PyTorch Tensors which enables heterogeneous-sized dimensions. \
    HetTensors can be constructed with nested list of torch.Tensor. HetTensor supports several operations, \
    including indexing and reduce operations (sum, mean, prod).",
)

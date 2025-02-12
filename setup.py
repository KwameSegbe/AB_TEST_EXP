# setup.py
from setuptools import setup

setup(
    name="ab_testing",
    version="0.1",
    packages=["ab_testing"],
    install_requires=["pandas", "scipy"],
)
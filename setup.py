"""Shim for editable installs with older pip/setuptools."""
from setuptools import setup, find_packages

setup(
    name="vsr-place",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)

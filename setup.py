from setuptools import find_packages, setup

import multisensorimport

setup(
    name='openarm-multisensor',
    version=multisensorimport.__version__,
    description="Muscle time series data analysis code.",
    author="Laura A Hallock",
    author_email="lhallock@eecs.berkeley.edu",
    url="https://simtk.org/projects/openarm",
    packages=find_packages(),
)

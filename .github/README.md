# ccrvam

> Python implementation of Checkerboard Copula Regression-based Visualization and Association Measure (CCRVAM)

[![PyPI version](https://badge.fury.io/py/ccrvam.svg)](https://badge.fury.io/py/ccrvam)
[![build](https://github.com/DhyeyMavani2003/ccrvam/actions/workflows/test.yaml/badge.svg)](https://github.com/DhyeyMavani2003/ccrvam/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/ccrvam/badge/?version=latest)](https://ccrvam.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/DhyeyMavani2003/ccrvam/badge.svg?branch=main)](https://coveralls.io/github/DhyeyMavani2003/ccrvam?branch=main)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.7.0-blue.svg)](https://github.com/christophevg/pypi-template)

## Authors

- Dhyey Mavani
- Daeyoung Kim 
- Shu-Min Liao

## Installation

This package (ccrvam) is hosted on PyPi, so for installation add the following line at the top of your Jupyter notebook!

```python
%pip install ccrvam
```

**Now, you should be all set to use it in a Jupyter Notebook!**

Alternatively, if you would like to use it in a project, we recommend you to have a virtual environment for your use of this package, then follow the following workflow. For best practices, it's recommended to use a virtual environment:

1. First, create and activate a virtual environment (Python 3.8+ recommended):

```bash
# Create virtual environment
$ python -m venv ccrvam-env

# Activate virtual environment (Mac/Linux)
$ source ccrvam-env/bin/activate

# Verify you're in the virtual environment
$ which python
```

2. Install package

```bash
$ pip install ccrvam
```

3. To deactivate the virtual environment, when done:

```bash
$ deactivate
```

## Documentation

Visit [Read the Docs](https://ccrvam.readthedocs.org) for the full documentation, including overviews and several examples.

## Examples

For detailed examples in Jupyter Notebooks and beyond (organized by functionality) please refer to our [GitHub repository's examples folder](https://github.com/DhyeyMavani2003/ccrvam/tree/master/examples).

## Features

- Construction of checkerboard copulas from contingency tables and/or list of cases
- Calculation of marginal distributions and CDFs
- Computation of Checkerboard Copula Regression (CCR) and Prediction based on CCR
- Implementation of Checkerboard Copula Regression Association Measure (CCRAM) and the Scaled CCRAM (SCCRAM)
- Bootstrap functionality for CCR-based prediction, CCRAM and SCCRAM
- Permutation testing functionality for CCRAM & SCCRAM
- Vectorized implementations for improved performance
- Rigorous Edge-case Handling & Unit Testing with Pytest 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite the following papers:

1. Zheng Wei and Daeyoung Kim (2021). On exploratory analytic method for multi-way contingency tables with an ordinal response variable and categorical explanatory variables. *Journal of Multivariate Analysis*, 186, 104793. https://doi.org/10.1016/j.jmva.2021.104793

2. Shu-Min Liao, Li Wang, Daeyoung Kim (2024). Visualization of Dependence in Multidimensional Contingency Tables with an Ordinal Dependent Variable via Copula Regression. In *Dependent Data in Social Sciences Research: Forms, Issues, and Methods of Analysis*, Second edition, Mark Stemmler, Wolfgang Wiedermann, and Francis L. Huang, eds. Springer New York LLC, pp. 517-538

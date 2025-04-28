# Getting Started with Installation

## Installation

### Usage in Jupyter Notebook:
This package (ccrvam) is hosted on PyPi, so for installation add the following line at the top of your Jupyter notebook!

```python3
%pip install --upgrade ccrvam
```

**Now, you should be all set to use it in a Jupyter Notebook!**

**Note:** You might need to restart your kernel after installation in order to use the package. This is because of the way Jupyter handles package installations.

### Usage in a Software Project Through Terminal (Virtual Environment):

Alternatively, if you would like to use it in a project, we recommend you to have a virtual environment for your use of this package, then follow the following workflow:

1. First, create and activate a virtual environment (Python 3.10+ recommended):

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

## Documentation Structure

Visit [Read the Docs](https://ccrvam.readthedocs.org) for the full documentation, including overviews and several examples.

## Examples

For detailed examples in Jupyter Notebooks and beyond (organized by functionality) please refer to our [GitHub repository's examples folder](https://github.com/DhyeyMavani2003/ccrvam/tree/master/examples).

## Features

- Construction of CCRVAM objects from three forms of categorical data (case form, frequency form, table form)
- Calculation of marginal distributions and CDFs of categorical variables
- Computation of Checkerboard Copula Regression (CCR), its Prediction and Visualization
- Implementation of Checkerboard Copula Regression Association Measure (CCRAM) and Scaled CCRAM (SCCRAM)
- Bootstrap functionality for CCR-based prediction, CCRAM and SCCRAM
- Permutation testing functionality for CCRAM & SCCRAM
- Vectorized implementations for improved performance
- Rigorous Edge-case Handling & Unit Testing with Pytest 

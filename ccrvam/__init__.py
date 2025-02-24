"""
  Python implementation of Checkerboard Copula Regression-based Visualization and Association Measure
"""
from ccrvam.checkerboard.gencopula import GenericCheckerboardCopula
from ccrvam.checkerboard.utils import gen_contingency_to_case_form, gen_case_form_to_contingency
from ccrvam.checkerboard.genstatsim import (
        bootstrap_ccram,
        bootstrap_predict_category_summary,
        permutation_test_ccram,
    )

__version__ = "0.3.0"
__all__ = [
  "GenericCheckerboardCopula",
  "gen_contingency_to_case_form",
  "gen_case_form_to_contingency",
  "bootstrap_ccram",
  "bootstrap_predict_category_summary",
  "permutation_test_ccram",
]
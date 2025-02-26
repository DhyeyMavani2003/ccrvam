"""
  Python implementation of Checkerboard Copula Regression-based Visualization and Association Measure
"""
from ccrvam.checkerboard.gencopula import GenericCCRVAM
from ccrvam.checkerboard.utils import gen_contingency_to_case_form, gen_case_form_to_contingency
from ccrvam.checkerboard.genstatsim import (
        bootstrap_ccram,
        bootstrap_predict_category_summary,
        permutation_test_ccram,
    )

__version__ = "0.4.0"
__all__ = [
  "GenericCCRVAM",
  "gen_contingency_to_case_form",
  "gen_case_form_to_contingency",
  "bootstrap_ccram",
  "bootstrap_predict_category_summary",
  "permutation_test_ccram",
]
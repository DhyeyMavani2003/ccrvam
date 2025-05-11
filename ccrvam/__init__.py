"""
  Python implementation of Checkerboard Copula Regression-based Visualization and Association Measure
"""
from ccrvam.checkerboard.genccrvam import GenericCCRVAM
from ccrvam.checkerboard.utils import DataProcessor, gen_contingency_to_case_form, gen_case_form_to_contingency
from ccrvam.checkerboard.genstatsim import (
    bootstrap_ccram,
    bootstrap_predict_ccr_summary,
    permutation_test_ccram,
    save_predictions,
)

__version__ = "1.1.2"
__all__ = [
  "GenericCCRVAM",
  "DataProcessor",
  "gen_contingency_to_case_form",
  "gen_case_form_to_contingency",
  "bootstrap_ccram",
  "bootstrap_predict_ccr_summary",
  "save_predictions",
  "permutation_test_ccram",
]
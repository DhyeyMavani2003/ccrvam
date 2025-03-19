import numpy as np
import pandas as pd
import pytest
from ccrvam import (
    bootstrap_ccram, 
    permutation_test_ccram, 
    bootstrap_predict_ccr_summary, 
)

@pytest.fixture
def contingency_table():
    """Fixture to create a sample contingency table."""
    return np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    
@pytest.fixture
def table_4d():
    """Fixture for 4D contingency table."""
    table = np.zeros((2,3,2,6), dtype=int)
    
    # RDA Row 1 [0,2,0,*]
    table[0,0,0,1] = 1
    table[0,0,0,4] = 2
    table[0,0,0,5] = 4
    
    # RDA Row 2 [0,2,1,*]
    table[0,0,1,3] = 1
    table[0,0,1,4] = 3
    
    # RDA Row 3 [0,1,0,*]
    table[0,1,0,1] = 2
    table[0,1,0,2] = 3
    table[0,1,0,4] = 6
    table[0,1,0,5] = 4
    
    # RDA Row 4 [0,1,1,*]
    table[0,1,1,1] = 1
    table[0,1,1,3] = 2
    table[0,1,1,5] = 1
    
    # RDA Row 5 [0,0,0,*]
    table[0,2,0,4] = 2 
    table[0,2,0,5] = 2
    
    # RDA Row 6 [0,0,1,*]
    table[0,2,1,2] = 1
    table[0,2,1,3] = 1
    table[0,2,1,4] = 3
    
    # RDA Row 7 [1,2,0,*]
    table[1,0,0,2] = 3
    table[1,0,0,4] = 1
    table[1,0,0,5] = 2
    
    # RDA Row 8 [1,2,1,*]
    table[1,0,1,1] = 1
    table[1,0,1,4] = 3
    
    # RDA Row 9 [1,1,0,*]
    table[1,1,0,1] = 3
    table[1,1,0,2] = 4
    table[1,1,0,3] = 5
    table[1,1,0,4] = 6
    table[1,1,0,5] = 2
    
    # RDA Row 10 [1,1,1,*]
    table[1,1,1,0] = 1
    table[1,1,1,1] = 4
    table[1,1,1,2] = 4
    table[1,1,1,3] = 3
    table[1,1,1,5] = 1
    
    # RDA Row 11 [1,0,0,*]
    table[1,2,0,0] = 2
    table[1,2,0,1] = 2
    table[1,2,0,2] = 1
    table[1,2,0,3] = 5
    table[1,2,0,4] = 2
    
    # RDA Row 12 [1,0,1,*]
    table[1,2,1,0] = 2
    table[1,2,1,2] = 2
    table[1,2,1,3] = 3
    
    return table

@pytest.fixture
def cases_4d():
    """Fixture for 4D case-form data in 1-indexed format."""
    return np.array([
        # RDA Row 1
        [1,1,1,2],[1,1,1,5],[1,1,1,5],
        [1,1,1,6],[1,1,1,6],[1,1,1,6],[1,1,1,6],
        # RDA Row 2
        [1,1,2,4],[1,1,2,5],[1,1,2,5],[1,1,2,5],
        # RDA Row 3
        [1,2,1,2],[1,2,1,2],[1,2,1,3],[1,2,1,3],[1,2,1,3],
        [1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],
        [1,2,1,6],[1,2,1,6],[1,2,1,6],[1,2,1,6],
        # RDA Row 4
        [1,2,2,2],[1,2,2,4],[1,2,2,4],[1,2,2,6],
        # RDA Row 5
        [1,3,1,5],[1,3,1,5],[1,3,1,6],[1,3,1,6],
        # RDA Row 6
        [1,3,2,3],[1,3,2,4],[1,3,2,5],[1,3,2,5],[1,3,2,5],
        # RDA Row 7
        [2,1,1,3],[2,1,1,3],[2,1,1,3],[2,1,1,5],[2,1,1,6],[2,1,1,6],
        # RDA Row 8
        [2,1,2,2],[2,1,2,5],[2,1,2,5],[2,1,2,5],
        # RDA Row 9
        [2,2,1,2],[2,2,1,2],[2,2,1,2],[2,2,1,3],[2,2,1,3],[2,2,1,3],[2,2,1,3],
        [2,2,1,4],[2,2,1,4],[2,2,1,4],[2,2,1,4],[2,2,1,4],
        [2,2,1,5],[2,2,1,5],[2,2,1,5],[2,2,1,5],[2,2,1,5],[2,2,1,5],
        [2,2,1,6],[2,2,1,6],
        # RDA Row 10
        [2,2,2,1],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],
        [2,2,2,3],[2,2,2,3],[2,2,2,3],[2,2,2,3],
        [2,2,2,4],[2,2,2,4],[2,2,2,4],[2,2,2,6],
        # RDA Row 11
        [2,3,1,1],[2,3,1,1],[2,3,1,2],[2,3,1,2],[2,3,1,3],
        [2,3,1,4],[2,3,1,4],[2,3,1,4],[2,3,1,4],[2,3,1,4],
        [2,3,1,5],[2,3,1,5],
        # RDA Row 12
        [2,3,2,1],[2,3,2,1],[2,3,2,3],[2,3,2,3],
        [2,3,2,4],[2,3,2,4],[2,3,2,4]
    ])

def test_bootstrap_ccram_basic(contingency_table):
    """Test basic functionality of bootstrap_ccram."""
    result = bootstrap_ccram(
        contingency_table,
        predictors=[1],
        response=2,
        n_resamples=999,
        random_state=8990
    )
    
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "bootstrap_distribution")
    assert hasattr(result, "standard_error")
    assert hasattr(result, "histogram_fig")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    assert result.standard_error >= 0

def test_bootstrap_ccram_multiple_axes(table_4d):
    """Test bootstrap_ccram with multiple conditioning axes."""
    result = bootstrap_ccram(
        table_4d,
        predictors=[1, 4],
        response=2,
        n_resamples=999,
        random_state=8990
    )

    assert "(X1,X4) to X2" in result.metric_name
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    
    result_full = bootstrap_ccram(
        table_4d,
        predictors=[1, 2, 3],
        response=4,
        n_resamples=999,
        random_state=8990
    )

    assert "(X1,X2,X3) to X4" in result_full.metric_name
    assert hasattr(result_full, "confidence_interval")
    assert result_full.confidence_interval[0] < result_full.confidence_interval[1]
    
    result_2d_multi = bootstrap_ccram(
        table_4d,
        predictors=[1],
        response=2,
        n_resamples=999,
        random_state=8990
    )
    
    assert "(X1) to X2" in result_2d_multi.metric_name
    assert hasattr(result_2d_multi, "confidence_interval")
    assert result_2d_multi.confidence_interval[0] < result_2d_multi.confidence_interval[1]

def test_prediction_summary_multi(table_4d):
    """Test multi-dimensional prediction summary."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1","X2"],
        response=4,
        n_resamples=999,
        random_state=8990
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert np.all(summary_df >= 0)
    assert np.all(summary_df <= 100)
    
    summary_df_full = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2, 3],
        predictors_names=["X1","X2","X3"],
        response=4,
        n_resamples=999,
        random_state=8990
    )

    assert isinstance(summary_df_full, pd.DataFrame)
    assert np.all(summary_df_full >= 0)
    assert np.all(summary_df_full <= 100)

def test_display_prediction_summary_multi(table_4d):
    """Test display of multi-dimensional prediction summary."""
    
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["First", "Second"],
        response=4,
        response_name="Fourth",
        n_resamples=999,
        random_state=8990
    )
    
    assert isinstance(summary_df, pd.DataFrame)

def test_permutation_test_multiple_axes(table_4d):
    """Test permutation test with multiple conditioning axes."""
    result = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=999,
        random_state=8990
    )
    print(result.p_value)
    assert "(X1,X2) to X4" in result.metric_name
    assert hasattr(result, "p_value")
    assert 0 <= result.p_value <= 1
    assert len(result.null_distribution) == 999
    
    result_full = permutation_test_ccram(
        table_4d,
        predictors=[1, 2, 3],
        response=4,
        n_resamples=999,
        random_state=8990
    )
    print(result_full.p_value)
    assert "(X1,X2,X3) to X4" in result_full.metric_name
    assert hasattr(result_full, "p_value")
    assert 0 <= result_full.p_value <= 1
    assert len(result_full.null_distribution) == 999

def test_invalid_inputs_multi():
    """Test invalid inputs for multi-axis functionality."""
    valid_table = np.array([[10, 0], [0, 10]])
    # Test invalid axes combinations
    with pytest.raises(ValueError):
        bootstrap_ccram(valid_table, predictors=[3, 4], response=1)
    
    # Test duplicate axes
    with pytest.raises(IndexError):
        bootstrap_ccram(valid_table, predictors=[1, 1], response=2)

def test_reproducibility_multi(table_4d):
    """Test reproducibility with multiple axes."""
    result1 = bootstrap_ccram(
        table_4d,
        predictors=[1, 2],
        response=3,
        random_state=8990
    )
    
    result2 = bootstrap_ccram(
        table_4d,
        predictors=[1, 2],
        response=3,
        random_state=8990
    )
    
    np.testing.assert_array_almost_equal(
        result1.bootstrap_distribution,
        result2.bootstrap_distribution
    )
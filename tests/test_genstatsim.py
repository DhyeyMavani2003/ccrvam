import numpy as np
import pandas as pd
import pytest
from ccrvam import (
    bootstrap_ccram, 
    permutation_test_ccram, 
    save_predictions,
    bootstrap_predict_ccr_summary, 
)
from ccrvam.checkerboard.genstatsim import (
    CustomBootstrapResult,
    CustomPermutationResult,
)
import matplotlib.pyplot as plt

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

def test_custom_bootstrap_result_plotting():
    """Test plotting functionality of CustomBootstrapResult."""
    # Create a sample bootstrap result
    result = CustomBootstrapResult(
        metric_name="Test Metric",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=np.random.normal(0.5, 0.1, 1000),
        standard_error=0.1
    )
    
    # Test plotting with title
    fig = result.plot_distribution(title="Test Plot")
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    
    # Test plotting with degenerate distribution
    result_degen = CustomBootstrapResult(
        metric_name="Degenerate Metric",
        observed_value=0.5,
        confidence_interval=(0.5, 0.5),
        bootstrap_distribution=np.array([0.5] * 1000),
        standard_error=0.0
    )
    fig_degen = result_degen.plot_distribution()
    assert fig_degen is not None
    assert isinstance(fig_degen, plt.Figure)

def test_custom_permutation_result_plotting():
    """Test plotting functionality of CustomPermutationResult."""
    # Create a sample permutation result
    result = CustomPermutationResult(
        metric_name="Test Metric",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.random.normal(0.3, 0.1, 1000)
    )
    
    # Test plotting with title
    fig = result.plot_distribution(title="Test Plot")
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    
    # Test plotting with degenerate distribution
    result_degen = CustomPermutationResult(
        metric_name="Degenerate Metric",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.array([0.5] * 1000)
    )
    fig_degen = result_degen.plot_distribution()
    assert fig_degen is not None
    assert isinstance(fig_degen, plt.Figure)

def test_bootstrap_ccram_scaled(table_4d):
    """Test bootstrap_ccram with scaled option."""
    result = bootstrap_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        scaled=True,
        n_resamples=999,
        random_state=8990
    )
    
    assert "SCCRAM" in result.metric_name
    assert hasattr(result, "confidence_interval")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    
    # Compare with unscaled version
    result_unscaled = bootstrap_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        scaled=False,
        n_resamples=999,
        random_state=8990
    )
    
    assert "CCRAM" in result_unscaled.metric_name
    assert result.observed_value != result_unscaled.observed_value

def test_permutation_test_alternatives(table_4d):
    """Test permutation test with different alternative hypotheses."""
    # Test 'greater' alternative
    result_greater = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='greater',
        n_resamples=999,
        random_state=8990
    )
    assert result_greater.p_value >= 0
    assert result_greater.p_value <= 1
    
    # Test 'less' alternative
    result_less = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='less',
        n_resamples=999,
        random_state=8990
    )
    assert result_less.p_value >= 0
    assert result_less.p_value <= 1
    
    # Test 'two-sided' alternative
    result_two_sided = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='two-sided',
        n_resamples=999,
        random_state=8990
    )
    assert result_two_sided.p_value >= 0
    assert result_two_sided.p_value <= 1

def test_save_predictions(table_4d, tmp_path):
    """Test saving prediction results to different formats."""
    # Generate prediction summary
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1", "X2"],
        response=4,
        n_resamples=999,
        random_state=8990
    )
    
    # Test saving to CSV
    csv_path = tmp_path / "predictions.csv"
    save_predictions(summary_df, save_path=str(csv_path), format='csv')
    assert csv_path.exists()
    
    # Test saving to TXT
    txt_path = tmp_path / "predictions.txt"
    save_predictions(summary_df, save_path=str(txt_path), format='txt')
    assert txt_path.exists()
    
    # Test invalid format
    with pytest.raises(ValueError):
        save_predictions(summary_df, save_path=str(tmp_path / "invalid.xyz"), format='xyz')

def test_bootstrap_ccram_store_tables(table_4d):
    """Test bootstrap_ccram with store_tables option."""
    result = bootstrap_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=999,
        random_state=8990,
        store_tables=True
    )
    
    assert result.bootstrap_tables is not None
    assert result.bootstrap_tables.shape == (999,) + table_4d.shape
    assert np.all(result.bootstrap_tables >= 0)

def test_permutation_test_store_tables(table_4d):
    """Test permutation_test_ccram with store_tables option."""
    result = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=999,
        random_state=8990,
        store_tables=True
    )
    
    assert result.permutation_tables is not None
    assert result.permutation_tables.shape == (999,) + table_4d.shape
    assert np.all(result.permutation_tables >= 0)

def test_bootstrap_predict_ccr_summary_edge_cases(table_4d):
    """Test bootstrap_predict_ccr_summary with edge cases."""
    # Test with single predictor
    result_single = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=1,
        response=4,
        n_resamples=999,
        random_state=8990
    )
    assert isinstance(result_single, pd.DataFrame)
    
    # Test with all predictors
    result_all = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2, 3],
        response=4,
        n_resamples=999,
        random_state=8990
    )
    assert isinstance(result_all, pd.DataFrame)
    
    # Test with custom names
    result_custom = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["Var1", "Var2"],
        response=4,
        response_name="Target",
        n_resamples=999,
        random_state=8990
    )
    assert isinstance(result_custom, pd.DataFrame)
    assert "Target" in result_custom.columns[0]
    assert "Var1" in result_custom.index[0]

def test_bootstrap_predict_ccr_summary_plotting(table_4d):
    """Test plotting functionality of bootstrap_predict_ccr_summary."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1", "X2"],
        response=4,
        n_resamples=999,
        random_state=8990
    )
    
    # Test basic plotting
    fig, ax = summary_df.plot_predictions_summary()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    
    # Test plotting with different options
    fig, ax = summary_df.plot_predictions_summary(
        show_values=False,
        show_indep_line=False,
        cmap='Reds',
        figsize=(12, 8),
        plot_type='bubble'
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_invalid_inputs_permutation_test():
    """Test invalid inputs for permutation test."""
    valid_table = np.array([[10, 0], [0, 10]])
    
    # Test invalid alternative hypothesis
    with pytest.raises(ValueError):
        permutation_test_ccram(
            valid_table,
            predictors=[1],
            response=2,
            alternative='invalid'
        )
    
    # Test invalid axes
    with pytest.raises(ValueError):
        permutation_test_ccram(
            valid_table,
            predictors=[3],
            response=1
        )
    
    # Test duplicate axes
    with pytest.raises(IndexError):
        permutation_test_ccram(
            valid_table,
            predictors=[1, 1],
            response=2
        )

def test_bootstrap_predict_ccr_summary_parallel_options(table_4d):
    """Test bootstrap_predict_ccr_summary with different parallel processing options."""
    # Test with parallel processing enabled (default)
    result_parallel = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1", "X2"],
        response=4,
        n_resamples=999,
        random_state=8990,
        parallel=True
    )
    assert isinstance(result_parallel, pd.DataFrame)
    assert np.all(result_parallel >= 0)
    assert np.all(result_parallel <= 100)
    
    # Test with parallel processing disabled
    result_sequential = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1", "X2"],
        response=4,
        n_resamples=999,
        random_state=8990,
        parallel=False
    )
    assert isinstance(result_sequential, pd.DataFrame)
    assert np.all(result_sequential >= 0)
    assert np.all(result_sequential <= 100)
    
    # Verify that both methods produce similar results
    # Note: Results won't be exactly identical due to random sampling,
    # but they should be reasonably close
    pd.testing.assert_frame_equal(
        result_parallel.round(1), 
        result_sequential.round(1),
        check_exact=False,
        rtol=0.1  # Allow for 10% relative tolerance
    )
    
    # Test that predictions attribute is present and consistent
    assert hasattr(result_parallel, 'predictions')
    assert hasattr(result_sequential, 'predictions')
    pd.testing.assert_frame_equal(
        result_parallel.predictions,
        result_sequential.predictions
    )

def test_bootstrap_predict_ccr_summary_parallel_edge_cases(table_4d):
    """Test bootstrap_predict_ccr_summary parallel processing with edge cases."""
    # Test with very small number of resamples
    result_small = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=10,
        random_state=8990,
        parallel=True
    )
    assert isinstance(result_small, pd.DataFrame)
    
    # Test with single predictor
    result_single_pred = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=1,
        response=4,
        n_resamples=100,
        random_state=8990,
        parallel=True
    )
    assert isinstance(result_single_pred, pd.DataFrame)
    
    # Test with all predictors in parallel and sequential modes
    result_all_parallel = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2, 3],
        response=4,
        n_resamples=100,
        random_state=8990,
        parallel=True
    )
    result_all_sequential = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2, 3],
        response=4,
        n_resamples=100,
        random_state=8990,
        parallel=False
    )
    assert isinstance(result_all_parallel, pd.DataFrame)
    assert isinstance(result_all_sequential, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result_all_parallel.round(1),
        result_all_sequential.round(1),
        check_exact=False,
        rtol=0.1
    )
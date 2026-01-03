import numpy as np
import pandas as pd
import pytest
from ccrvam import (
    bootstrap_ccram, 
    permutation_test_ccram, 
    save_predictions,
    bootstrap_predict_ccr_summary,
    all_subsets_ccram,
    best_subset_ccram,
    SubsetCCRAMResult,
    BestSubsetCCRAMResult,
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
    
def test_bootstrap_ccram_parallel(contingency_table):
    """Test bootstrap_ccram with parallel option."""
    result = bootstrap_ccram(
        contingency_table,
        predictors=[1],
        response=2,
        n_resamples=999,
        method='percentile',
        parallel=True,
        random_state=8990
    )
    
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "bootstrap_distribution")
    assert hasattr(result, "standard_error")
    assert hasattr(result, "histogram_fig")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    assert result.standard_error >= 0
    
    result_basic = bootstrap_ccram(
        contingency_table,
        predictors=[1],
        response=2,
        n_resamples=999,
        method='basic',
        parallel=True,
        random_state=8990
    )
    
    assert hasattr(result_basic, "confidence_interval")
    assert hasattr(result_basic, "bootstrap_distribution")
    assert hasattr(result_basic, "standard_error")
    assert hasattr(result_basic, "histogram_fig")
    assert result_basic.confidence_interval[0] < result_basic.confidence_interval[1]
    assert result_basic.standard_error >= 0
    
    result_bca = bootstrap_ccram(
        contingency_table,
        predictors=[1],
        response=2,
        n_resamples=999,
        method='bca',
        parallel=True,
        random_state=8990
    )
    
    assert hasattr(result_bca, "confidence_interval")
    assert hasattr(result_bca, "bootstrap_distribution")
    assert hasattr(result_bca, "standard_error")
    assert hasattr(result_bca, "histogram_fig")
    assert result_bca.confidence_interval[0] < result_bca.confidence_interval[1]
    assert result_bca.standard_error >= 0
    
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

def test_bootstrap_ccram_parallel_options(table_4d):
    """Test bootstrap_ccram with parallel options."""
    result = bootstrap_ccram(
        table_4d,
        predictors=[1, 4],
        response=3,
        n_resamples=999,
        parallel=True,
        random_state=8990
    )
    
    assert hasattr(result, "confidence_interval")
    assert hasattr(result, "bootstrap_distribution")
    assert hasattr(result, "histogram_fig")
    assert result.confidence_interval[0] < result.confidence_interval[1]
    assert result.standard_error >= 0

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
    
def test_permutation_test_parallel_options(table_4d):
    """Test permutation_test_ccram with parallel options."""
    result = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='greater',
        parallel=True,
        n_resamples=999,
        random_state=8990
    )
    assert hasattr(result, "p_value")
    assert 0 <= result.p_value <= 1
    assert len(result.null_distribution) == 999
    
    result_less = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='less',
        parallel=True,
        n_resamples=999,
        random_state=8990
    )
    assert hasattr(result_less, "p_value")
    assert 0 <= result_less.p_value <= 1
    assert len(result_less.null_distribution) == 999
    
    result_two_sided = permutation_test_ccram(
        table_4d,
        predictors=[1, 2],
        response=4,
        alternative='two-sided',
        parallel=True,
        n_resamples=999,
        random_state=8990
    )
    assert hasattr(result_two_sided, "p_value")
    assert 0 <= result_two_sided.p_value <= 1
    assert len(result_two_sided.null_distribution) == 999

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

# New tests for plotting customization options

def test_bootstrap_result_plot_customization():
    """Test font size and figure customization in bootstrap result plots."""
    result = CustomBootstrapResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=np.random.normal(0.5, 0.1, 1000),
        standard_error=0.1
    )
    
    # Test all customization options
    fig = result.plot_distribution(
        title="Custom Bootstrap Distribution",
        figsize=(12, 8),
        title_fontsize=16,
        xlabel_fontsize=14,
        ylabel_fontsize=12,
        tick_fontsize=10,
        text_fontsize=8
    )
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == 12
    assert fig.get_size_inches()[1] == 8
    plt.close(fig)

def test_bootstrap_result_plot_kwargs():
    """Test **kwargs functionality in bootstrap result plots."""
    result = CustomBootstrapResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=np.random.normal(0.5, 0.1, 1000),
        standard_error=0.1
    )
    
    # Test with matplotlib kwargs
    fig = result.plot_distribution(
        facecolor='lightgray',
        edgecolor='black',
        alpha=0.9
    )
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_bootstrap_result_plot_degenerate_with_customization():
    """Test degenerate case with text font customization."""
    result = CustomBootstrapResult(
        metric_name="Degenerate CCRAM",
        observed_value=0.5,
        confidence_interval=(0.5, 0.5),
        bootstrap_distribution=np.array([0.5] * 1000),
        standard_error=0.0
    )
    
    # Test degenerate case with custom text font size
    fig = result.plot_distribution(
        title="Degenerate Distribution",
        figsize=(10, 6),
        title_fontsize=18,
        text_fontsize=12
    )
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_permutation_result_plot_customization():
    """Test font size and figure customization in permutation result plots."""
    result = CustomPermutationResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.random.normal(0.3, 0.1, 1000)
    )
    
    # Test all customization options (no text_fontsize for permutation plots)
    fig = result.plot_distribution(
        title="Custom Null Distribution",
        figsize=(14, 10),
        title_fontsize=20,
        xlabel_fontsize=16,
        ylabel_fontsize=14,
        tick_fontsize=12
    )
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == 14
    assert fig.get_size_inches()[1] == 10
    plt.close(fig)

def test_permutation_result_plot_kwargs():
    """Test **kwargs functionality in permutation result plots."""
    result = CustomPermutationResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.random.normal(0.3, 0.1, 1000)
    )
    
    # Test with matplotlib kwargs
    fig = result.plot_distribution(
        facecolor='white',
        edgecolor='gray',
        linewidth=2
    )
    
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_prediction_summary_plot_customization_heatmap(table_4d):
    """Test heatmap plotting with all customization options."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["X1", "X2"],
        response=4,
        response_name="Response",
        n_resamples=100,
        random_state=8990
    )
    
    # Test heatmap with all customization options
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='heatmap',
        figsize=(16, 12),
        title_fontsize=18,
        xlabel_fontsize=14,
        ylabel_fontsize=14,
        tick_fontsize=12,
        text_fontsize=10,
        use_category_letters=False,
        show_values=True,
        show_indep_line=True,
        cmap='Blues'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert fig.get_size_inches()[0] == 16
    assert fig.get_size_inches()[1] == 12
    plt.close(fig)

def test_prediction_summary_plot_customization_bubble(table_4d):
    """Test bubble plotting with all customization options."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["Var1", "Var2"],
        response=4,
        response_name="Target",
        n_resamples=100,
        random_state=8990
    )
    
    # Test bubble plot with all customization options
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='bubble',
        figsize=(14, 10),
        title_fontsize=16,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        tick_fontsize=10,
        text_fontsize=8,
        use_category_letters=True,
        show_values=False,
        show_indep_line=False,
        cmap='Reds'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

def test_prediction_summary_plot_category_letters(table_4d):
    """Test category letters functionality in prediction summary plots."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        predictors_names=["First", "Second"],
        response=4,
        response_name="Fourth",
        n_resamples=100,
        random_state=8990
    )
    
    # Test with category letters enabled
    fig, ax = summary_df.plot_predictions_summary(
        use_category_letters=True,
        plot_type='heatmap'
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with category letters enabled for bubble plot
    fig, ax = summary_df.plot_predictions_summary(
        use_category_letters=True,
        plot_type='bubble'
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_prediction_summary_plot_kwargs(table_4d):
    """Test **kwargs functionality in prediction summary plots."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=100,
        random_state=8990
    )
    
    # Test with matplotlib kwargs
    fig, ax = summary_df.plot_predictions_summary(
        facecolor='lightblue',
        edgecolor='navy',
        alpha=0.8
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

def test_prediction_summary_plot_save_with_customization(table_4d, tmp_path):
    """Test saving prediction summary plots with customization."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=100,
        random_state=8990
    )
    
    # Test saving heatmap with custom options
    heatmap_path = tmp_path / "custom_heatmap.png"
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='heatmap',
        figsize=(12, 8),
        title_fontsize=16,
        use_category_letters=True,
        save_path=str(heatmap_path),
        dpi=150
    )
    
    assert heatmap_path.exists()
    plt.close(fig)
    
    # Test saving bubble plot
    bubble_path = tmp_path / "custom_bubble.png"
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='bubble',
        figsize=(10, 8),
        title_fontsize=14,
        use_category_letters=False,
        save_path=str(bubble_path),
        dpi=200
    )
    
    assert bubble_path.exists()
    plt.close(fig)

def test_plot_customization_none_values():
    """Test that None values for font sizes use default behavior."""
    # Bootstrap result
    result = CustomBootstrapResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=np.random.normal(0.5, 0.1, 100),
        standard_error=0.1
    )
    
    fig = result.plot_distribution(
        title_fontsize=None,
        xlabel_fontsize=None,
        ylabel_fontsize=None,
        tick_fontsize=None,
        text_fontsize=None
    )
    
    assert fig is not None
    plt.close(fig)
    
    # Permutation result
    perm_result = CustomPermutationResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.random.normal(0.3, 0.1, 100)
    )
    
    fig = perm_result.plot_distribution(
        title_fontsize=None,
        xlabel_fontsize=None,
        ylabel_fontsize=None,
        tick_fontsize=None
    )
    
    assert fig is not None
    plt.close(fig)

def test_plot_customization_edge_cases(table_4d):
    """Test edge cases for plot customization."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1],
        response=4,
        n_resamples=50,
        random_state=8990
    )
    
    # Test with very small figure size
    fig, ax = summary_df.plot_predictions_summary(
        figsize=(4, 3),
        title_fontsize=8,
        tick_fontsize=6
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with very large figure size
    fig, ax = summary_df.plot_predictions_summary(
        figsize=(20, 15),
        title_fontsize=24,
        tick_fontsize=18
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_customization_invalid_plot_type(table_4d):
    """Test error handling for invalid plot types."""
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=50,
        random_state=8990
    )
    
    # Test invalid plot type
    with pytest.raises(ValueError, match="plot_type must be either 'heatmap' or 'bubble'"):
        summary_df.plot_predictions_summary(plot_type='invalid')

def test_bootstrap_result_plot_missing_distribution():
    """Test bootstrap result plotting when bootstrap_distribution is None."""
    result = CustomBootstrapResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=None,  # Missing distribution
        standard_error=0.1
    )
    
    # Should return None and print warning
    fig = result.plot_distribution()
    assert fig is None

def test_plot_customization_backward_compatibility(table_4d):
    """Test that all plotting functions work without new parameters."""
    # Test bootstrap result plotting (old style)
    result = CustomBootstrapResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        confidence_interval=(0.3, 0.7),
        bootstrap_distribution=np.random.normal(0.5, 0.1, 100),
        standard_error=0.1
    )
    
    fig = result.plot_distribution()  # No new parameters
    assert fig is not None
    plt.close(fig)
    
    # Test permutation result plotting (old style)
    perm_result = CustomPermutationResult(
        metric_name="Test CCRAM",
        observed_value=0.5,
        p_value=0.05,
        null_distribution=np.random.normal(0.3, 0.1, 100)
    )
    
    fig = perm_result.plot_distribution()  # No new parameters
    assert fig is not None
    plt.close(fig)
    
    # Test prediction summary plotting (old style)
    summary_df = bootstrap_predict_ccr_summary(
        table_4d,
        predictors=[1, 2],
        response=4,
        n_resamples=50,
        random_state=8990
    )
    
    fig, ax = summary_df.plot_predictions_summary()  # No new parameters
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# =============================================================================
# Tests for zero-count combinations (NA predictions)
# =============================================================================

@pytest.fixture
def table_with_zero_count_combinations():
    """Fixture for a 3D table with zero-count predictor combinations."""
    # When (X1=1, X2=1), there are no observations (all zeros across X3)
    return np.array([
        [[0, 0, 0],   # X1=1, X2=1, X3=1,2,3 -> all zeros (no data)
         [5, 2, 3]],  # X1=1, X2=2, X3=1,2,3
        [[4, 1, 0],   # X1=2, X2=1, X3=1,2,3
         [2, 3, 5]]   # X1=2, X2=2, X3=1,2,3
    ])


def test_bootstrap_predict_ccr_summary_zero_count_predictions(table_with_zero_count_combinations):
    """Test that bootstrap_predict_ccr_summary returns NaN for zero-count combinations."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Check that predictions attribute exists
    assert hasattr(summary_df, 'predictions')
    
    # Check that the first column (X1=1, X2=1) has NaN prediction
    first_col = 'X1=1 X2=1'
    pred_value = summary_df.predictions.loc[first_col, 'Predicted']
    assert pd.isna(pred_value), f"Zero-count combination should have NaN prediction, got {pred_value}"
    
    # Check that other combinations have valid predictions
    for col in ['X1=1 X2=2', 'X1=2 X2=1', 'X1=2 X2=2']:
        pred = summary_df.predictions.loc[col, 'Predicted']
        assert not pd.isna(pred), f"Non-zero combination {col} should have valid prediction"


def test_bootstrap_predict_ccr_summary_zero_count_percentages(table_with_zero_count_combinations):
    """Test that bootstrap percentages are NaN for zero-count combinations and sum to 100 for valid ones."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Check that summary_df has values for all combinations
    assert len(summary_df) == 4  # 2x2 predictor combinations
    
    # Zero-count combination (X1=1, X2=1) should have NaN percentages
    zero_count_col = 'X1=1 X2=1'
    assert summary_df.loc[zero_count_col].isna().all(), \
        f"Zero-count combination {zero_count_col} should have all NaN percentages"
    
    # Non-zero-count combinations should have percentages that sum to ~100
    non_zero_cols = ['X1=1 X2=2', 'X1=2 X2=1', 'X1=2 X2=2']
    for col in non_zero_cols:
        col_sum = summary_df.loc[col].sum()
        assert abs(col_sum - 100.0) < 0.01, f"Column {col} should sum to ~100%, got {col_sum}"


def test_plot_predictions_summary_zero_count_heatmap(table_with_zero_count_combinations):
    """Test heatmap plotting with zero-count combinations."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Should not raise any error
    fig, ax = summary_df.plot_predictions_summary(plot_type='heatmap')
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_predictions_summary_zero_count_bubble(table_with_zero_count_combinations):
    """Test bubble plotting with zero-count combinations."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Should not raise any error
    fig, ax = summary_df.plot_predictions_summary(plot_type='bubble')
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_predictions_summary_zero_count_with_save(table_with_zero_count_combinations):
    """Test saving plot with zero-count combinations."""
    import tempfile
    import os
    
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'zero_count_summary.png')
        
        fig, ax = summary_df.plot_predictions_summary(
            plot_type='heatmap',
            save_path=save_path,
            dpi=100
        )
        
        # Verify file was saved
        assert os.path.exists(save_path)
        plt.close(fig)


def test_plot_predictions_summary_zero_count_with_category_letters(table_with_zero_count_combinations):
    """Test plotting with category letters and zero-count combinations."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Test with category letters
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='heatmap',
        use_category_letters=True
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_predictions_zero_count_csv(table_with_zero_count_combinations):
    """Test saving predictions with zero-count combinations to CSV."""
    import tempfile
    import os
    
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'zero_count_predictions.csv')
        
        save_predictions(summary_df, save_path=save_path, format='csv')
        
        # Verify file was saved
        assert os.path.exists(save_path)
        
        # Read and verify content
        with open(save_path, 'r') as f:
            content = f.read()
        
        # Check that NA is present for zero-count combination
        assert 'NA' in content, "CSV should contain 'NA' for zero-count combinations"


def test_save_predictions_zero_count_txt(table_with_zero_count_combinations):
    """Test saving predictions with zero-count combinations to TXT."""
    import tempfile
    import os
    
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'zero_count_predictions.txt')
        
        save_predictions(summary_df, save_path=save_path, format='txt')
        
        # Verify file was saved
        assert os.path.exists(save_path)
        
        # Read and verify content
        with open(save_path, 'r') as f:
            content = f.read()
        
        # Check that NA is present for zero-count combination
        assert 'NA' in content, "TXT should contain 'NA' for zero-count combinations"


def test_bootstrap_predict_ccr_summary_multiple_zero_count():
    """Test handling of multiple zero-count combinations."""
    # Table with multiple zero-count combinations
    table = np.array([
        [[0, 0, 0],   # X1=1, X2=1 -> zero count
         [0, 0, 0]],  # X1=1, X2=2 -> zero count
        [[4, 1, 0],   # X1=2, X2=1 -> has data
         [2, 3, 5]]   # X1=2, X2=2 -> has data
    ])
    
    summary_df = bootstrap_predict_ccr_summary(
        table,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # First two combinations should have NaN predictions
    assert pd.isna(summary_df.predictions.loc['X1=1 X2=1', 'Predicted'])
    assert pd.isna(summary_df.predictions.loc['X1=1 X2=2', 'Predicted'])
    
    # Last two combinations should have valid predictions
    assert not pd.isna(summary_df.predictions.loc['X1=2 X2=1', 'Predicted'])
    assert not pd.isna(summary_df.predictions.loc['X1=2 X2=2', 'Predicted'])


def test_plot_predictions_summary_zero_count_show_indep_line(table_with_zero_count_combinations):
    """Test plotting with independence line and zero-count combinations."""
    summary_df = bootstrap_predict_ccr_summary(
        table_with_zero_count_combinations,
        predictors=[1, 2],
        predictors_names=['X1', 'X2'],
        response=3,
        response_name='X3',
        n_resamples=100,
        random_state=42,
        parallel=False
    )
    
    # Test with show_indep_line=True
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='heatmap',
        show_indep_line=True
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with show_indep_line=False
    fig, ax = summary_df.plot_predictions_summary(
        plot_type='bubble',
        show_indep_line=False
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# =============================================================================
# Tests for all_subsets_ccram and best_subset_ccram
# =============================================================================

@pytest.fixture
def table_2d():
    """Fixture for 2D contingency table."""
    return np.array([
        [10, 5, 2],
        [3, 15, 8],
        [1, 4, 12]
    ])


@pytest.fixture
def table_3d():
    """Fixture for 3D contingency table."""
    return np.array([
        [[5, 2], [3, 8]],
        [[2, 7], [6, 4]],
        [[1, 3], [4, 9]]
    ])


class TestAllSubsetsCCRAM:
    """Tests for all_subsets_ccram function."""
    
    def test_basic_2d_ccram(self, table_2d):
        """Test basic functionality with 2D table and CCRAM."""
        result = all_subsets_ccram(table_2d, response=2, scaled=False)
        
        assert isinstance(result, SubsetCCRAMResult)
        assert isinstance(result.results_df, pd.DataFrame)
        assert result.response == 2
        assert result.n_dimensions == 2
        assert result.scaled is False
        assert result.metric_column == 'ccram'
        
        # Check DataFrame columns
        assert 'k' in result.results_df.columns
        assert 'predictors' in result.results_df.columns
        assert 'pred_cate' in result.results_df.columns
        assert 'response' in result.results_df.columns
        assert 'ccram' in result.results_df.columns
        assert 'sccram' not in result.results_df.columns
        
        # For 2D table with response=2, only k=1 with predictor X1 is possible
        assert len(result.results_df) == 1
        assert result.results_df.iloc[0]['k'] == 1
        assert result.results_df.iloc[0]['predictors'] == (1,)
        # table_2d is 3x3, so X1 has 3 categories
        assert result.results_df.iloc[0]['pred_cate'] == (3,)
    
    def test_basic_2d_sccram(self, table_2d):
        """Test basic functionality with 2D table and SCCRAM."""
        result = all_subsets_ccram(table_2d, response=2, scaled=True)
        
        assert result.scaled is True
        assert result.metric_column == 'sccram'
        assert 'sccram' in result.results_df.columns
        assert 'ccram' not in result.results_df.columns
    
    def test_basic_3d_ccram(self, table_3d):
        """Test basic functionality with 3D table and CCRAM."""
        result = all_subsets_ccram(table_3d, response=3, scaled=False)
        
        assert result.n_dimensions == 3
        assert result.scaled is False
        
        # For 3D table with response=3: k=1 has 2 subsets (X1, X2), k=2 has 1 subset (X1,X2)
        assert len(result.results_df) == 3
        
        # Check k=1 subsets
        k1_subsets = result.results_df[result.results_df['k'] == 1]
        assert len(k1_subsets) == 2
        
        # Check k=2 subsets
        k2_subsets = result.results_df[result.results_df['k'] == 2]
        assert len(k2_subsets) == 1
        assert k2_subsets.iloc[0]['predictors'] == (1, 2)
    
    def test_basic_4d_ccram(self, table_4d):
        """Test basic functionality with 4D table and CCRAM."""
        result = all_subsets_ccram(table_4d, response=4, scaled=False)
        
        assert result.n_dimensions == 4
        assert result.scaled is False
        
        # For 4D table with response=4:
        # k=1: 3 subsets (X1, X2, X3)
        # k=2: 3 subsets (X1,X2), (X1,X3), (X2,X3)
        # k=3: 1 subset (X1,X2,X3)
        # Total: 7 subsets
        assert len(result.results_df) == 7
        
        k1_subsets = result.results_df[result.results_df['k'] == 1]
        k2_subsets = result.results_df[result.results_df['k'] == 2]
        k3_subsets = result.results_df[result.results_df['k'] == 3]
        
        assert len(k1_subsets) == 3
        assert len(k2_subsets) == 3
        assert len(k3_subsets) == 1
    
    def test_basic_4d_sccram(self, table_4d):
        """Test basic functionality with 4D table and SCCRAM."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        assert result.scaled is True
        assert result.metric_column == 'sccram'
        assert 'sccram' in result.results_df.columns
        
        # Values should be different from CCRAM
        result_ccram = all_subsets_ccram(table_4d, response=4, scaled=False)
        
        # Compare same predictor combination
        sccram_val = result.results_df[result.results_df['predictors'] == (1, 2, 3)]['sccram'].values[0]
        ccram_val = result_ccram.results_df[result_ccram.results_df['predictors'] == (1, 2, 3)]['ccram'].values[0]
        
        # SCCRAM and CCRAM should be different (unless variance is 1)
        assert sccram_val != ccram_val or np.isclose(sccram_val, ccram_val, rtol=0.01)
    
    def test_4d_with_specific_k(self, table_4d):
        """Test with specific k value in 4D table."""
        # Test k=1
        result_k1 = all_subsets_ccram(table_4d, response=4, k=1)
        assert len(result_k1.results_df) == 3
        assert all(result_k1.results_df['k'] == 1)
        
        # Test k=2
        result_k2 = all_subsets_ccram(table_4d, response=4, k=2)
        assert len(result_k2.results_df) == 3
        assert all(result_k2.results_df['k'] == 2)
        
        # Test k=3
        result_k3 = all_subsets_ccram(table_4d, response=4, k=3)
        assert len(result_k3.results_df) == 1
        assert all(result_k3.results_df['k'] == 3)
    
    def test_4d_different_response_axes(self, table_4d):
        """Test with different response axes in 4D table."""
        # Response = X1
        result_r1 = all_subsets_ccram(table_4d, response=1)
        assert result_r1.response == 1
        assert len(result_r1.results_df) == 7  # k=1: 3, k=2: 3, k=3: 1
        
        # Predictors should be X2, X3, X4
        all_preds = set()
        for preds in result_r1.results_df['predictors']:
            all_preds.update(preds)
        assert all_preds == {2, 3, 4}
        
        # Response = X2
        result_r2 = all_subsets_ccram(table_4d, response=2)
        assert result_r2.response == 2
        
        # Predictors should be X1, X3, X4
        all_preds = set()
        for preds in result_r2.results_df['predictors']:
            all_preds.update(preds)
        assert all_preds == {1, 3, 4}
    
    def test_with_variable_names(self, table_4d):
        """Test with custom variable names."""
        var_names = {1: 'Age', 2: 'Income', 3: 'Education', 4: 'Satisfaction'}
        result = all_subsets_ccram(table_4d, response=4, variable_names=var_names)
        
        assert 'predictor_names' in result.results_df.columns
        
        # Check that names are correctly assigned
        row_with_all = result.results_df[result.results_df['k'] == 3].iloc[0]
        assert row_with_all['predictor_names'] == '(Age, Income, Education)'
    
    def test_sorting_within_k(self, table_4d):
        """Test that results are sorted by metric descending within each k."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Check sorting within k=1
        k1_values = result.results_df[result.results_df['k'] == 1]['sccram'].values
        assert all(k1_values[i] >= k1_values[i+1] for i in range(len(k1_values)-1))
        
        # Check sorting within k=2
        k2_values = result.results_df[result.results_df['k'] == 2]['sccram'].values
        assert all(k2_values[i] >= k2_values[i+1] for i in range(len(k2_values)-1))
    
    def test_get_top_subsets(self, table_4d):
        """Test get_top_subsets method with renamed 'top' parameter."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Test with keyword argument 'top'
        top_3 = result.get_top_subsets(top=3)
        assert len(top_3) == 3
        
        # Should be sorted by sccram descending
        assert all(top_3['sccram'].values[i] >= top_3['sccram'].values[i+1] 
                   for i in range(len(top_3)-1))
        
        # Test getting more than available
        top_10 = result.get_top_subsets(top=10)
        assert len(top_10) == 7  # Only 7 subsets available
        
        # Test with positional argument (backward compatibility)
        top_2 = result.get_top_subsets(2)
        assert len(top_2) == 2
    
    def test_get_top_subsets_per_k(self, table_4d):
        """Test get_top_subsets_per_k method for getting top subsets for each k."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Get top 2 subsets for each k
        top_per_k = result.get_top_subsets_per_k(top=2)
        
        # For 4D table with response=4:
        # k=1: 3 possible subsets (X1, X2, X3) -> returns top 2
        # k=2: 3 possible pairs -> returns top 2
        # k=3: 1 possible triplet -> returns 1 (less than requested 2)
        assert len(top_per_k) == 5  # 2 + 2 + 1 = 5
        
        # Verify we have correct number for each k
        assert len(top_per_k[top_per_k['k'] == 1]) == 2
        assert len(top_per_k[top_per_k['k'] == 2]) == 2
        assert len(top_per_k[top_per_k['k'] == 3]) == 1
        
        # Verify sorting within each k (descending by sccram)
        for k in [1, 2]:
            k_data = top_per_k[top_per_k['k'] == k]['sccram'].values
            assert all(k_data[i] >= k_data[i+1] for i in range(len(k_data)-1))
    
    def test_get_top_subsets_per_k_exceeds_available(self, table_4d):
        """Test get_top_subsets_per_k when top exceeds available combinations."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Request top 10, but k=1 has only 3, k=2 has only 3, k=3 has only 1
        top_per_k = result.get_top_subsets_per_k(top=10)
        
        # Should return all available: 3 + 3 + 1 = 7
        assert len(top_per_k) == 7
        assert len(top_per_k[top_per_k['k'] == 1]) == 3
        assert len(top_per_k[top_per_k['k'] == 2]) == 3
        assert len(top_per_k[top_per_k['k'] == 3]) == 1
    
    def test_get_top_subsets_per_k_default(self, table_4d):
        """Test get_top_subsets_per_k with default top=3."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Default is top=3
        top_per_k = result.get_top_subsets_per_k()
        
        # k=1: 3 available, returns 3
        # k=2: 3 available, returns 3  
        # k=3: 1 available, returns 1
        assert len(top_per_k) == 7
    
    def test_get_subsets_by_k(self, table_4d):
        """Test get_subsets_by_k method."""
        result = all_subsets_ccram(table_4d, response=4)
        
        k2_subsets = result.get_subsets_by_k(2)
        assert len(k2_subsets) == 3
        assert all(k2_subsets['k'] == 2)
    
    def test_summary_ccram(self, table_4d):
        """Test summary method with CCRAM."""
        result = all_subsets_ccram(table_4d, response=4, scaled=False)
        summary = result.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'k' in summary.columns
        assert 'max_ccram' in summary.columns
        assert 'mean_ccram' in summary.columns
        assert 'min_ccram' in summary.columns
        assert 'n_subsets' in summary.columns
        
        # Check n_subsets
        assert summary[summary['k'] == 1]['n_subsets'].values[0] == 3
        assert summary[summary['k'] == 2]['n_subsets'].values[0] == 3
        assert summary[summary['k'] == 3]['n_subsets'].values[0] == 1
    
    def test_summary_sccram(self, table_4d):
        """Test summary method with SCCRAM."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        summary = result.summary()
        
        assert 'max_sccram' in summary.columns
        assert 'mean_sccram' in summary.columns
        assert 'min_sccram' in summary.columns
        assert 'max_ccram' not in summary.columns
    
    def test_invalid_response_axis(self, table_4d):
        """Test error handling for invalid response axis."""
        with pytest.raises(ValueError, match="out of bounds"):
            all_subsets_ccram(table_4d, response=5)
        
        with pytest.raises(ValueError, match="out of bounds"):
            all_subsets_ccram(table_4d, response=0)
    
    def test_invalid_k_value(self, table_4d):
        """Test error handling for invalid k value."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            all_subsets_ccram(table_4d, response=4, k=0)
        
        with pytest.raises(ValueError, match="k must be less than"):
            all_subsets_ccram(table_4d, response=4, k=4)
    
    def test_ccram_values_are_valid(self, table_4d):
        """Test that CCRAM values are within valid range [0, 1]."""
        result = all_subsets_ccram(table_4d, response=4, scaled=False)
        
        assert all(result.results_df['ccram'] >= 0)
        assert all(result.results_df['ccram'] <= 1)
    
    def test_sccram_values_are_valid(self, table_4d):
        """Test that SCCRAM values are within valid range [0, 1]."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        assert all(result.results_df['sccram'] >= 0)
        assert all(result.results_df['sccram'] <= 1)
    
    def test_pred_cate_column_4d(self, table_4d):
        """Test pred_cate column values for 4D table."""
        # table_4d has shape (2, 3, 2, 6)
        result = all_subsets_ccram(table_4d, response=4, scaled=False)
        
        # Check that pred_cate column exists
        assert 'pred_cate' in result.results_df.columns
        
        # Check specific values for k=1 predictors
        # X1 has 2 categories, X2 has 3, X3 has 2
        k1_df = result.results_df[result.results_df['k'] == 1]
        
        x1_row = k1_df[k1_df['predictors'] == (1,)]
        assert len(x1_row) == 1
        assert x1_row.iloc[0]['pred_cate'] == (2,)
        
        x2_row = k1_df[k1_df['predictors'] == (2,)]
        assert len(x2_row) == 1
        assert x2_row.iloc[0]['pred_cate'] == (3,)
        
        x3_row = k1_df[k1_df['predictors'] == (3,)]
        assert len(x3_row) == 1
        assert x3_row.iloc[0]['pred_cate'] == (2,)
        
        # Check k=2 predictor combinations
        k2_df = result.results_df[result.results_df['k'] == 2]
        
        x12_row = k2_df[k2_df['predictors'] == (1, 2)]
        assert len(x12_row) == 1
        assert x12_row.iloc[0]['pred_cate'] == (2, 3)
        
        x13_row = k2_df[k2_df['predictors'] == (1, 3)]
        assert len(x13_row) == 1
        assert x13_row.iloc[0]['pred_cate'] == (2, 2)
        
        x23_row = k2_df[k2_df['predictors'] == (2, 3)]
        assert len(x23_row) == 1
        assert x23_row.iloc[0]['pred_cate'] == (3, 2)
        
        # Check k=3 predictor combination
        k3_df = result.results_df[result.results_df['k'] == 3]
        x123_row = k3_df[k3_df['predictors'] == (1, 2, 3)]
        assert len(x123_row) == 1
        assert x123_row.iloc[0]['pred_cate'] == (2, 3, 2)
    
    def test_pred_cate_is_tuple_for_computation(self, table_4d):
        """Test that pred_cate is a tuple that can be used directly for computation."""
        result = all_subsets_ccram(table_4d, response=4, scaled=False)
        
        # pred_cate should be a tuple, not a string
        k3_row = result.results_df[result.results_df['k'] == 3].iloc[0]
        pred_cate = k3_row['pred_cate']
        
        assert isinstance(pred_cate, tuple)
        assert pred_cate == (2, 3, 2)
        
        # Verify we can compute sum and product directly without parsing
        assert sum(pred_cate) == 7
        assert np.prod(pred_cate) == 12
        
        # Verify this works with apply() for creating new columns
        df = result.results_df.copy()
        df['sum_cate'] = df['pred_cate'].apply(sum)
        df['prod_cate'] = df['pred_cate'].apply(np.prod)
        
        # Check computed values
        k3_computed = df[df['k'] == 3].iloc[0]
        assert k3_computed['sum_cate'] == 7
        assert k3_computed['prod_cate'] == 12
    
    def test_get_results_with_penalties(self, table_4d):
        """Test get_results_with_penalties method returns correct penalty columns."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True, 
                                   variable_names={1: 'A', 2: 'B', 3: 'C', 4: 'D'})
        
        df = result.get_results_with_penalties()
        
        # Check that penalty columns exist
        assert 'sum_cate' in df.columns
        assert 'prod_cate' in df.columns
        
        # Check that original columns are preserved
        assert 'k' in df.columns
        assert 'predictors' in df.columns
        assert 'pred_cate' in df.columns
        assert 'response' in df.columns
        assert 'sccram' in df.columns
        assert 'predictor_names' in df.columns
        
        # predictors column should be present (now a tuple, not string)
        assert 'predictors' in df.columns
        
        # Verify specific values for k=1 predictors (table_4d shape is (2, 3, 2, 6))
        # X1 has 2 categories, X2 has 3, X3 has 2
        k1_df = df[df['k'] == 1]
        
        x1_row = k1_df[k1_df['predictors'] == (1,)].iloc[0]
        assert x1_row['sum_cate'] == 2
        assert x1_row['prod_cate'] == 2
        
        x2_row = k1_df[k1_df['predictors'] == (2,)].iloc[0]
        assert x2_row['sum_cate'] == 3
        assert x2_row['prod_cate'] == 3
        
        # Verify k=2 values
        k2_df = df[df['k'] == 2]
        
        x12_row = k2_df[k2_df['predictors'] == (1, 2)].iloc[0]
        assert x12_row['sum_cate'] == 5  # 2 + 3
        assert x12_row['prod_cate'] == 6  # 2 * 3
        
        # Verify k=3 values
        k3_df = df[df['k'] == 3]
        x123_row = k3_df.iloc[0]
        assert x123_row['sum_cate'] == 7   # 2 + 3 + 2
        assert x123_row['prod_cate'] == 12  # 2 * 3 * 2


class TestBestSubsetCCRAM:
    """Tests for best_subset_ccram function."""
    
    def test_basic_2d_ccram(self, table_2d):
        """Test basic functionality with 2D table and CCRAM."""
        result = best_subset_ccram(table_2d, response=2, scaled=False)
        
        assert isinstance(result, BestSubsetCCRAMResult)
        assert result.response == 2
        assert result.scaled is False
        assert result.predictors == (1,)
        assert result.k == 1
        assert result.ccram >= 0
        assert result.rank_within_k == 1
        assert result.total_subsets_in_k == 1
    
    def test_basic_2d_sccram(self, table_2d):
        """Test basic functionality with 2D table and SCCRAM."""
        result = best_subset_ccram(table_2d, response=2, scaled=True)
        
        assert result.scaled is True
        assert result.ccram >= 0
    
    def test_basic_4d_ccram(self, table_4d):
        """Test basic functionality with 4D table and CCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=False)
        
        assert isinstance(result, BestSubsetCCRAMResult)
        assert result.response == 4
        assert result.scaled is False
        assert result.ccram >= 0
        assert isinstance(result.all_results, SubsetCCRAMResult)
        
        # Best subset should have highest CCRAM
        max_ccram = result.all_results.results_df['ccram'].max()
        assert result.ccram == max_ccram
    
    def test_basic_4d_sccram(self, table_4d):
        """Test basic functionality with 4D table and SCCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=True)
        
        assert result.scaled is True
        
        # Best subset should have highest SCCRAM
        max_sccram = result.all_results.results_df['sccram'].max()
        assert result.ccram == max_sccram
    
    def test_4d_with_specific_k(self, table_4d):
        """Test with specific k value in 4D table."""
        # Test k=1
        result_k1 = best_subset_ccram(table_4d, response=4, k=1)
        assert result_k1.k == 1
        assert len(result_k1.predictors) == 1
        
        # Test k=2
        result_k2 = best_subset_ccram(table_4d, response=4, k=2)
        assert result_k2.k == 2
        assert len(result_k2.predictors) == 2
        
        # Test k=3
        result_k3 = best_subset_ccram(table_4d, response=4, k=3)
        assert result_k3.k == 3
        assert len(result_k3.predictors) == 3
        assert result_k3.predictors == (1, 2, 3)
    
    def test_repr_ccram(self, table_4d):
        """Test __repr__ method with CCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=False)
        repr_str = repr(result)
        
        assert 'BestSubsetCCRAMResult' in repr_str
        assert 'CCRAM:' in repr_str
        assert 'SCCRAM:' not in repr_str
        assert 'Predictors:' in repr_str
        assert 'Response: X4' in repr_str
    
    def test_repr_sccram(self, table_4d):
        """Test __repr__ method with SCCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=True)
        repr_str = repr(result)
        
        assert 'SCCRAM:' in repr_str
    
    def test_summary_df_ccram(self, table_4d):
        """Test summary_df method with CCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=False)
        summary = result.summary_df()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'metric' in summary.columns
        assert 'value' in summary.columns
        
        metrics = summary['metric'].tolist()
        assert 'CCRAM' in metrics
        assert 'SCCRAM' not in metrics
    
    def test_summary_df_sccram(self, table_4d):
        """Test summary_df method with SCCRAM."""
        result = best_subset_ccram(table_4d, response=4, scaled=True)
        summary = result.summary_df()
        
        metrics = summary['metric'].tolist()
        assert 'SCCRAM' in metrics
        assert 'CCRAM' not in metrics
    
    def test_with_variable_names(self, table_4d):
        """Test with custom variable names."""
        var_names = {1: 'Age', 2: 'Income', 3: 'Education', 4: 'Satisfaction'}
        result = best_subset_ccram(table_4d, response=4, variable_names=var_names)
        
        # Variable names should be in all_results
        assert 'predictor_names' in result.all_results.results_df.columns
    
    def test_rank_within_k(self, table_4d):
        """Test rank_within_k is correctly computed."""
        result = best_subset_ccram(table_4d, response=4)
        
        # The best subset should have rank 1 within its k
        assert result.rank_within_k >= 1
        assert result.rank_within_k <= result.total_subsets_in_k
    
    def test_all_results_access(self, table_4d):
        """Test that all_results provides full access to subset data."""
        result = best_subset_ccram(table_4d, response=4)
        
        all_results = result.all_results
        assert isinstance(all_results, SubsetCCRAMResult)
        assert len(all_results.results_df) == 7
    
    def test_invalid_response_axis(self, table_4d):
        """Test error handling for invalid response axis."""
        with pytest.raises(ValueError, match="out of bounds"):
            best_subset_ccram(table_4d, response=5)
    
    def test_invalid_k_value(self, table_4d):
        """Test error handling for invalid k value."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            best_subset_ccram(table_4d, response=4, k=0)


class TestSubsetCCRAMConsistency:
    """Tests for consistency between all_subsets_ccram and best_subset_ccram."""
    
    def test_best_matches_all_subsets(self, table_4d):
        """Test that best_subset_ccram returns the same best as all_subsets_ccram."""
        all_result = all_subsets_ccram(table_4d, response=4)
        best_result = best_subset_ccram(table_4d, response=4)
        
        # Find best from all_result
        best_from_all = all_result._results_df_full.loc[all_result._results_df_full['ccram'].idxmax()]
        
        assert best_result.predictors == best_from_all['predictors']
        assert best_result.ccram == best_from_all['ccram']
    
    def test_scaled_consistency(self, table_4d):
        """Test consistency between scaled=True and scaled=False."""
        result_ccram = all_subsets_ccram(table_4d, response=4, scaled=False)
        result_sccram = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        # Same number of subsets
        assert len(result_ccram.results_df) == len(result_sccram.results_df)
        
        # Same predictor combinations (in some order)
        ccram_preds = set(result_ccram.results_df['predictors'].tolist())
        sccram_preds = set(result_sccram.results_df['predictors'].tolist())
        assert ccram_preds == sccram_preds
    
    def test_k_restriction_consistency(self, table_4d):
        """Test that k restriction gives subset of full results."""
        full_result = all_subsets_ccram(table_4d, response=4)
        k2_result = all_subsets_ccram(table_4d, response=4, k=2)
        
        # k=2 results should be subset of full results
        full_k2 = full_result.results_df[full_result.results_df['k'] == 2]
        
        assert len(k2_result.results_df) == len(full_k2)
        
        # Values should match
        for _, row in k2_result.results_df.iterrows():
            matching = full_k2[full_k2['predictors'] == row['predictors']]
            assert len(matching) == 1
            assert matching.iloc[0]['ccram'] == row['ccram']


class TestSubsetCCRAMEdgeCases:
    """Tests for edge cases in subset CCRAM functions."""
    
    def test_2d_table_single_predictor(self, table_2d):
        """Test with 2D table where only one predictor is possible."""
        result = all_subsets_ccram(table_2d, response=1)
        
        assert len(result.results_df) == 1
        assert result.results_df.iloc[0]['predictors'] == (2,)
        assert result.results_df.iloc[0]['k'] == 1
    
    def test_deterministic_table(self):
        """Test with a deterministic (perfectly predictable) table."""
        # Perfect relationship: X1 completely determines X2
        deterministic_table = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])
        
        result = all_subsets_ccram(deterministic_table, response=2, scaled=True)
        
        # SCCRAM should be close to 1 for perfect relationship
        assert result.results_df.iloc[0]['sccram'] > 0.9
    
    def test_independent_table(self):
        """Test with an independent (uniform) table."""
        # Uniform distribution - no relationship
        independent_table = np.ones((3, 3), dtype=int) * 10
        
        result = all_subsets_ccram(independent_table, response=2, scaled=False)
        
        # CCRAM should be close to 0 for independent relationship
        assert result.results_df.iloc[0]['ccram'] < 0.1
    
    def test_large_k_4d(self, table_4d):
        """Test that k cannot exceed ndim-1."""
        # k=3 is maximum for 4D table
        result = all_subsets_ccram(table_4d, response=4, k=3)
        assert len(result.results_df) == 1
        
        # k=4 should raise error
        with pytest.raises(ValueError):
            all_subsets_ccram(table_4d, response=4, k=4)
    
    def test_metric_column_property(self, table_4d):
        """Test metric_column property returns correct column name."""
        result_ccram = all_subsets_ccram(table_4d, response=4, scaled=False)
        result_sccram = all_subsets_ccram(table_4d, response=4, scaled=True)
        
        assert result_ccram.metric_column == 'ccram'
        assert result_sccram.metric_column == 'sccram'
    
    def test_variable_names_partial(self, table_4d):
        """Test with partial variable names (some missing)."""
        partial_names = {1: 'Var1', 3: 'Var3'}  # Missing 2 and 4
        result = all_subsets_ccram(table_4d, response=4, variable_names=partial_names)
        
        # Should use default names for missing ones
        row = result.results_df[result.results_df['predictors'] == (1, 2, 3)].iloc[0]
        assert row['predictor_names'] == '(Var1, X2, Var3)'
    
    def test_response_in_middle(self, table_4d):
        """Test with response variable in middle position."""
        result = all_subsets_ccram(table_4d, response=2)
        
        # Predictors should be X1, X3, X4 (not X2)
        all_preds = set()
        for preds in result.results_df['predictors']:
            all_preds.update(preds)
        assert 2 not in all_preds
        assert all_preds == {1, 3, 4}


class TestPlotSubsets:
    """Tests for the plot_subsets method in SubsetCCRAMResult."""
    
    def test_plot_subsets_basic(self, table_4d):
        """Test basic plot_subsets functionality."""
        result = all_subsets_ccram(table_4d, response=4)
        fig, ax = result.plot_subsets()
        
        assert fig is not None
        assert ax is not None
        assert plt.get_fignums()  # Check figure was created
        plt.close('all')
    
    def test_plot_subsets_scaled(self, table_4d):
        """Test plot_subsets with scaled CCRAM."""
        result = all_subsets_ccram(table_4d, response=4, scaled=True)
        fig, ax = result.plot_subsets()
        
        assert fig is not None
        # Check title contains SCCRAM
        assert 'SCCRAM' in ax.get_title()
        plt.close('all')
    
    def test_plot_subsets_custom_figsize(self, table_4d):
        """Test plot_subsets with custom figure size."""
        result = all_subsets_ccram(table_4d, response=4)
        custom_figsize = (12, 8)
        fig, ax = result.plot_subsets(figsize=custom_figsize)
        
        np.testing.assert_array_almost_equal(fig.get_size_inches(), custom_figsize)
        plt.close('all')
    
    def test_plot_subsets_font_customization(self, table_4d):
        """Test plot_subsets font size customization."""
        result = all_subsets_ccram(table_4d, response=4)
        fig, ax = result.plot_subsets(
            title_fontsize=16,
            xlabel_fontsize=14,
            ylabel_fontsize=12,
            tick_fontsize=10,
            label_fontsize=9
        )
        
        assert fig is not None
        plt.close('all')
    
    def test_plot_subsets_custom_title(self, table_4d):
        """Test plot_subsets with custom title."""
        result = all_subsets_ccram(table_4d, response=4)
        custom_title = "My Custom Plot Title"
        fig, ax = result.plot_subsets(title=custom_title)
        
        assert ax.get_title() == custom_title
        plt.close('all')
    
    def test_plot_subsets_point_customization(self, table_4d):
        """Test plot_subsets point customization."""
        result = all_subsets_ccram(table_4d, response=4)
        fig, ax = result.plot_subsets(
            point_size=100,
            point_color='red'
        )
        
        assert fig is not None
        plt.close('all')
    
    def test_plot_subsets_save(self, table_4d, tmp_path):
        """Test plot_subsets save functionality."""
        result = all_subsets_ccram(table_4d, response=4)
        save_path = tmp_path / "test_plot.png"
        
        fig, ax = result.plot_subsets(save_path=str(save_path), dpi=100)
        
        assert save_path.exists()
        plt.close('all')
    
    def test_plot_subsets_save_creates_directory(self, table_4d, tmp_path):
        """Test plot_subsets creates directory if needed."""
        result = all_subsets_ccram(table_4d, response=4)
        save_path = tmp_path / "new_subdir" / "test_plot.png"
        
        fig, ax = result.plot_subsets(save_path=str(save_path))
        
        assert save_path.exists()
        assert save_path.parent.exists()
        plt.close('all')
    
    def test_plot_subsets_2d_table(self, table_2d):
        """Test plot_subsets with 2D table (single k value)."""
        result = all_subsets_ccram(table_2d, response=1)
        fig, ax = result.plot_subsets()
        
        assert fig is not None
        # Only k=1 should be present
        assert len(result.results_df['k'].unique()) == 1
        plt.close('all')
    
    def test_plot_subsets_3d_table(self, table_3d):
        """Test plot_subsets with 3D table."""
        result = all_subsets_ccram(table_3d, response=3)
        fig, ax = result.plot_subsets()
        
        assert fig is not None
        # k=1 and k=2 should be present
        assert set(result.results_df['k'].unique()) == {1, 2}
        plt.close('all')
    
    def test_plot_subsets_kwargs(self, table_4d):
        """Test plot_subsets with additional kwargs."""
        result = all_subsets_ccram(table_4d, response=4)
        fig, ax = result.plot_subsets(
            facecolor='lightgray'
        )
        
        assert fig is not None
        plt.close('all')
    
    def test_plot_subsets_specific_k(self, table_4d):
        """Test plot_subsets when results are for specific k only."""
        result = all_subsets_ccram(table_4d, response=4, k=2)
        fig, ax = result.plot_subsets()
        
        assert fig is not None
        # Only k=2 should be present
        assert set(result.results_df['k'].unique()) == {2}
        plt.close('all')
    
    def test_plot_subsets_backward_compatibility(self, table_4d):
        """Test that plot_subsets works with minimal arguments."""
        result = all_subsets_ccram(table_4d, response=4)
        
        # Should work with no arguments
        fig, ax = result.plot_subsets()
        assert fig is not None
        assert ax is not None
        plt.close('all')
    
    def test_plot_subsets_different_responses(self, table_4d):
        """Test plot_subsets with different response variables."""
        for response in [1, 2, 3, 4]:
            result = all_subsets_ccram(table_4d, response=response)
            fig, ax = result.plot_subsets()
            
            assert f'X{response}' in ax.get_title()
            plt.close('all')
    
    def test_plot_subsets_has_labels(self, table_4d):
        """Test that plot_subsets shows labels for best subsets per k."""
        result = all_subsets_ccram(table_4d, response=4)
        fig, ax = result.plot_subsets()
        
        # Check that annotations exist (one for each k)
        annotations = [child for child in ax.get_children() 
                      if isinstance(child, plt.Annotation)]
        k_values = result.results_df['k'].unique()
        assert len(annotations) == len(k_values)
        plt.close('all')
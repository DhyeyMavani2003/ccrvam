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
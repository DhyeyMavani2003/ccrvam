import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import os 
import tempfile
from ccrvam import GenericCCRVAM

@pytest.fixture
def generic_ccrvam():
    """Fixture providing a GenericCCRVAM instance with test data."""
    P = np.array([
        [0, 0, 2/8],
        [0, 1/8, 0],
        [2/8, 0, 0],
        [0, 1/8, 0],
        [0, 0, 2/8]
    ])
    return GenericCCRVAM(P)

@pytest.fixture
def contingency_table():
    """Fixture providing a test contingency table."""
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
    """
    Fixture for 4D case-form data 0-indexed here for utils 
    because they are supposed to be internal converter functions.
    """
    return np.array([
        # RDA Row 1
        [0,0,0,1],[0,0,0,4],[0,0,0,4],
        [0,0,0,5], [0,0,0,5],[0,0,0,5],[0,0,0,5],
        # RDA Row 2
        [0,0,1,3],[0,0,1,4],[0,0,1,4],[0,0,1,4],
        # RDA Row 3
        [0,1,0,1],[0,1,0,1],[0,1,0,2],[0,1,0,2],[0,1,0,2],
        [0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],[0,1,0,4],
        [0,1,0,5],[0,1,0,5],[0,1,0,5],[0,1,0,5],
        # RDA Row 4
        [0,1,1,1],[0,1,1,3],[0,1,1,3],[0,1,1,5],
        # RDA Row 5
        [0,2,0,4],[0,2,0,4],[0,2,0,5],[0,2,0,5],
        # RDA Row 6
        [0,2,1,2],[0,2,1,3],[0,2,1,4],[0,2,1,4],[0,2,1,4],
        # RDA Row 7
        [1,0,0,2],[1,0,0,2],[1,0,0,2],[1,0,0,4],[1,0,0,5],[1,0,0,5],
        # RDA Row 8
        [1,0,1,1],[1,0,1,4],[1,0,1,4],[1,0,1,4],
        # RDA Row 9
        [1,1,0,1],[1,1,0,1],[1,1,0,1],[1,1,0,2],[1,1,0,2],[1,1,0,2],[1,1,0,2],
        [1,1,0,3],[1,1,0,3],[1,1,0,3],[1,1,0,3],[1,1,0,3],
        [1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],[1,1,0,4],
        [1,1,0,5],[1,1,0,5],
        # RDA Row 10
        [1,1,1,0],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],
        [1,1,1,2],[1,1,1,2],[1,1,1,2],[1,1,1,2],
        [1,1,1,3],[1,1,1,3],[1,1,1,3],[1,1,1,5],
        # RDA Row 11
        [1,2,0,0],[1,2,0,0],[1,2,0,1],[1,2,0,1],[1,2,0,2],
        [1,2,0,3],[1,2,0,3],[1,2,0,3],[1,2,0,3],[1,2,0,3],
        [1,2,0,4],[1,2,0,4],
        # RDA Row 12
        [1,2,1,0],[1,2,1,0],[1,2,1,2],[1,2,1,2],
        [1,2,1,3],[1,2,1,3],[1,2,1,3]
    ])
    
@pytest.fixture
def expected_shape():
    """Fixture providing expected shape for the ccrvam."""
    return (2, 3, 2, 6)

# Basic Creation Tests
def test_from_contingency_table_valid(contingency_table):
    """Test valid contingency table initialization."""
    ccrvam = GenericCCRVAM.from_contingency_table(contingency_table)
    expected_P = contingency_table / contingency_table.sum()
    np.testing.assert_array_almost_equal(ccrvam.P, expected_P)

@pytest.mark.parametrize("invalid_table,error_msg", [
    (np.array([[1, 2], [3, -1]]), "Contingency table cannot contain negative values"),
    (np.array([[0, 0], [0, 0]]), "Contingency table cannot be all zeros"),
])
def test_invalid_contingency_tables(invalid_table, error_msg):
    """Test error handling for invalid contingency tables."""
    with pytest.raises(ValueError, match=error_msg):
        GenericCCRVAM.from_contingency_table(invalid_table)

# Marginal Distribution Tests
@pytest.mark.parametrize("expected_cdf_0, expected_cdf_1", [
    ([0, 2/8, 3/8, 5/8, 6/8, 1], [0, 2/8, 4/8, 1])
])
def test_marginal_cdfs(generic_ccrvam, expected_cdf_0, expected_cdf_1):
    """Test marginal CDF calculations."""
    np.testing.assert_almost_equal(generic_ccrvam.marginal_cdfs[0], expected_cdf_0)
    np.testing.assert_almost_equal(generic_ccrvam.marginal_cdfs[1], expected_cdf_1)

# Conditional PMF Tests
def test_conditional_pmfs(generic_ccrvam):
    """Test conditional PMF calculations."""
    expected_1_given_0 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    pmf, _ = generic_ccrvam._calculate_conditional_pmf(1, [0])
    np.testing.assert_array_almost_equal(pmf, expected_1_given_0)

# Regression Tests
@pytest.mark.parametrize("given_values, given_axes, target_axis, expected_value", [
    # Single conditioning
    ([0.0], [0], 1, 12/16),
    ([3/8], [0], 1, 6/16),
    ([1.0], [0], 1, 12/16),
])
def test_calculate_regression(generic_ccrvam, given_values, given_axes, target_axis, expected_value):
    """Test regression calculation with multiple conditioning axes."""
    calculated = generic_ccrvam._calculate_regression_batched(
        target_axis=target_axis,
        given_axes=given_axes,
        given_values=given_values
    )
    np.testing.assert_almost_equal(calculated, expected_value)

# CCRAM Tests
@pytest.mark.parametrize("predictors, response, expected_ccram", [
    ([1], 2, 0.84375),          # Single axis X1->X2
    ([2], 1, 0.0),              # Single axis X2->X1
])
def test_calculate_CCRAM(generic_ccrvam, predictors, response, expected_ccram):
    """Test CCRAM calculations with multiple conditioning axes."""
    calculated = generic_ccrvam.calculate_CCRAM(predictors, response, scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)

# SCCRAM Tests
@pytest.mark.parametrize("predictors, response, expected_sccram", [
    ([1], 2, 0.84375/(12*0.0703125)),          # Single axis X1->X2
    ([2], 1, 0.0),                             # Single axis X2->X1
])
def test_calculate_SCCRAM(generic_ccrvam, predictors, response, expected_sccram):
    """Test SCCRAM calculations with multiple conditioning axes."""
    calculated = generic_ccrvam.calculate_CCRAM(predictors, response, scaled=True)
    np.testing.assert_almost_equal(calculated, expected_sccram)

# Category Prediction Tests
@pytest.mark.parametrize("source_categories, predictors, response, expected_categories", [
    # Single axis prediction
    ([0], [0], 1, [2]),
    ([1], [0], 1, [1]),
    ([2], [0], 1, [0]),
    ([3], [0], 1, [1]),
    ([4], [0], 1, [2]),     
])
def test_predict_category_multi(generic_ccrvam, source_categories, predictors, response, expected_categories):
    """Test category prediction with multiple conditioning axes."""
    predicted = generic_ccrvam._predict_category_batched_multi(
        source_categories=source_categories,
        predictors=predictors,
        response=response
    )
    np.testing.assert_array_equal(predicted, expected_categories)

# Add Multi-axis Category Predictions Test
def test_get_predictions_ccr(generic_ccrvam):
    """Test category predictions with multiple conditioning axes."""
    df = generic_ccrvam.get_predictions_ccr(
        predictors=[2],
        response=1,
        variable_names={1: "Income", 2: "Education"}
    )
    
    assert isinstance(df, pd.DataFrame)
    assert "Predicted Income Category" in df.columns
    assert "Education Category" in df.columns
    
    df_nonames = generic_ccrvam.get_predictions_ccr(
        predictors=[2],
        response=1
    )
    
    assert isinstance(df_nonames, pd.DataFrame)
    assert "Predicted Response Category" in df_nonames.columns
    assert "X2 Category" in df_nonames.columns

# Add Consistency Tests for Multi-axis
def test_multi_axis_consistency(generic_ccrvam):
    """Test consistency between single and multiple axis calculations."""
    single_axis = generic_ccrvam.calculate_CCRAM(1, 2)
    multi_axis = generic_ccrvam.calculate_CCRAM([1], 2)
    np.testing.assert_almost_equal(single_axis, multi_axis)

# Invalid Cases Tests
def test_invalid_predictions(generic_ccrvam):
    """Test invalid prediction handling."""
    with pytest.raises(IndexError):
        generic_ccrvam._predict_category(5, 0, 1)

# Special Cases Tests
def test_prediction_special_cases(generic_ccrvam):
    """Test edge cases in predictions."""
    single_pred = generic_ccrvam._predict_category_batched_multi(np.array([0]), 0, 1)
    assert len(single_pred) == 1
    assert single_pred[0] == generic_ccrvam._predict_category(0, 0, 1)

def test_get_prediction_under_indep(generic_ccrvam):
    """Test prediction under independence."""
    # For the first axis (responses in the 1st dimension)
    pred_cat = generic_ccrvam.get_prediction_under_indep(1)
    assert isinstance(pred_cat, (int, np.int64))
    assert 1 <= pred_cat <= generic_ccrvam.P.shape[0]
    
    expected_cat = 3
    assert pred_cat == expected_cat
    
    # For the second axis (responses in the 2nd dimension)
    pred_cat_ax2 = generic_ccrvam.get_prediction_under_indep(2)
    assert isinstance(pred_cat_ax2, (int, np.int64))
    assert 1 <= pred_cat_ax2 <= generic_ccrvam.P.shape[1]

    expected_cat_ax2 = 2
    assert pred_cat_ax2 == expected_cat_ax2

# Consistency Tests
def test_calculation_consistency(contingency_table):
    """Test consistency across different initialization methods."""
    P = contingency_table / contingency_table.sum()
    cop1 = GenericCCRVAM(P)
    cop2 = GenericCCRVAM.from_contingency_table(contingency_table)
    
    np.testing.assert_array_almost_equal(
        cop1.calculate_CCRAM(1, 2),
        cop2.calculate_CCRAM(1, 2)
    )
        
def test_calculate_ccs_valid(generic_ccrvam):
    """Test valid calculation of scores."""
    scores_1 = generic_ccrvam.calculate_ccs(1)
    scores_2 = generic_ccrvam.calculate_ccs(2)

    # Check exact expected values
    expected_scores_1 = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875], dtype=np.float64)
    expected_scores_2 = np.array([0.125, 0.375, 0.75], dtype=np.float64)
    
    np.testing.assert_array_almost_equal(scores_1, expected_scores_1)
    np.testing.assert_array_almost_equal(scores_2, expected_scores_2)
    
def test_calculate_ccs_invalid_axis(generic_ccrvam):
    """Test invalid axis handling for score calculation."""
    with pytest.raises(KeyError):
        generic_ccrvam.calculate_ccs(3)  # Invalid axis index

def test_calculate_variance_ccs_valid(generic_ccrvam):
    """Test valid calculation of score variance."""
    var_1 = generic_ccrvam.calculate_variance_ccs(1)
    var_2 = generic_ccrvam.calculate_variance_ccs(2)
    
    # Check return type
    assert isinstance(var_1, (float, np.float64))
    assert isinstance(var_2, (float, np.float64))
    
    # Variance should be non-negative
    assert var_1 >= 0
    assert var_2 >= 0
    
    # Check exact expected values
    expected_var_1, expected_var_2 = 0.0791015625, 0.0703125
    np.testing.assert_almost_equal(var_1, expected_var_1)
    np.testing.assert_almost_equal(var_2, expected_var_2)

def test_calculate_variance_ccs_invalid_axis(generic_ccrvam):
    """Test invalid axis handling for variance calculation."""
    with pytest.raises(KeyError):
        generic_ccrvam.calculate_variance_ccs(3)  # Invalid axis index

def test_from_cases_creation(cases_4d, table_4d, expected_shape):
    """Test creation of ccrvam from cases data."""
    # Upping cases by 1 to account for 1-indexing
    cop = GenericCCRVAM.from_cases(cases_4d+1, expected_shape)
    assert cop.ndim == 4
    assert cop.P.shape == expected_shape
    assert np.all(cop.P >= 0)
    assert np.isclose(cop.P.sum(), 1.0)
    assert np.all(cop.P == GenericCCRVAM.from_contingency_table(table_4d).P)

def test_from_cases_marginal_pmfs(cases_4d, expected_shape):
    """Test marginal PMFs calculation from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test marginal PMFs exist for each dimension
    assert len(cop.marginal_pmfs) == 4
    
    # Test each marginal PMF sums to 1
    for axis in range(4):
        pmf = cop.marginal_pmfs[axis]
        assert np.isclose(np.sum(pmf), 1.0)

def test_from_cases_marginal_cdfs(cases_4d, expected_shape):
    """Test marginal CDFs calculation from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test CDFs exist for each dimension
    assert len(cop.marginal_cdfs) == 4
    
    # Test CDF properties
    for axis in range(4):
        cdf = cop.marginal_cdfs[axis]
        assert cdf[0] == 0  # CDF starts at 0
        assert np.isclose(cdf[-1], 1.0)  # CDF ends at 1
        assert np.all(np.diff(cdf) >= 0)  # CDF is monotonically increasing

def test_from_cases_scores(cases_4d, expected_shape):
    """Test scores calculation from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test scores exist for each dimension
    for axis in range(4):
        scores = cop.calculate_ccs(axis+1)
        assert len(scores) == expected_shape[axis]

def test_from_cases_variance(cases_4d, expected_shape):
    """Test variance calculation from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test variance for the last dimension
    variance = cop.calculate_variance_ccs(4)
    assert isinstance(variance, (float, np.float64))
    print(variance * 12)
    assert variance >= 0
    assert np.isclose(variance * 12, 0.9585)

def test_from_cases_invalid_input(cases_4d):
    """Test error handling for invalid inputs."""
    invalid_cases = np.array([[0,1], [1,2]])  # Wrong number of dimensions
    invalid_shape = (2, 2)  # Wrong shape specification
    
    with pytest.raises(ValueError):
        GenericCCRVAM.from_cases(invalid_cases, (2,2,2,2))
    
    with pytest.raises(ValueError):
        GenericCCRVAM.from_cases(cases_4d, invalid_shape)
    
def test_4d_ccram_calculations(cases_4d, expected_shape):
    """Test CCRAM calculations for 4D case with multiple conditioning axes."""
    # Upping cases by 1 to account for 1-indexing
    cop = GenericCCRVAM.from_cases(cases_4d+1, expected_shape)
    
    # Test various axis combinations
    test_cases = [
        ([1], 2, 0.015414243190773603),
        ([2], 1, 0.01597020761808525),
        ([4], 3, 0.04996195517713011),
        ([4], 2, 0.12290134029717903),
        ([1, 2], 3, 0.024360372462467715),
        ([1, 2, 3], 4, 0.25756048821793687),
        ([2, 3, 4], 1, 0.24583094094416508),
        ([3, 4], 2, 0.17679609577731298),
        ([1, 4], 2, 0.15210990526069565),
        ([1, 3, 4], 2, 0.2534201412985138),
        ([1, 2, 4], 3, 0.2682552130743483)
    ]
    
    for predictors, response, expected in test_cases:
        # Regular CCRAM
        ccram = cop.calculate_CCRAM(predictors, response, scaled=False)
        assert 0 <= ccram <= 1
        assert np.isclose(ccram, expected)

def test_4d_full_sccram_calculations(cases_4d, expected_shape):
    # Upping cases by 1 to account for 1-indexing
    cop = GenericCCRVAM.from_cases(cases_4d+1, expected_shape)
    sccram = cop.calculate_CCRAM([1,2,3], 4, scaled=True)
    assert 0 <= sccram <= 1
    assert np.isclose(sccram, 0.26870972725631526)

def test_4d_prediction_multi(cases_4d, expected_shape):
    """Test multi-axis prediction for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test various prediction scenarios
    test_cases = [
        # source_categories, predictors, response
        ([0], [0], 1),
        ([0, 1], [0, 1], 2),
        ([0, 1, 0], [0, 1, 2], 3)
    ]
    
    for source_cats, predictors, response in test_cases:
        predicted = cop._predict_category_batched_multi(
            source_categories=source_cats,
            predictors=predictors,
            response=response
        )
        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == (1,)
        assert 0 <= predicted[0] < expected_shape[response]

def test_4d_conditional_pmf(cases_4d, expected_shape):
    """Test conditional PMF calculations for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test various conditioning combinations
    test_cases = [
        (0, [1]),             # Single conditioning
        (0, [1, 3])           # Double conditioning
    ]
    
    for target, given_axes in test_cases:
        pmf, _ = cop._calculate_conditional_pmf(target, given_axes)
        assert isinstance(pmf, np.ndarray)

def test_4d_scores_expected_values(cases_4d, expected_shape):
    """Test score calculations for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    for axis in range(4):
        scores = cop.calculate_ccs(axis+1)
        assert len(scores) == expected_shape[axis]
        assert np.all(0 <= np.array(scores)) and np.all(np.array(scores) <= 1)
        # Scores should be monotonically increasing
        assert np.all(np.diff(scores) >= 0)

def test_get_prediction_under_indep_4d(cases_4d, expected_shape):
    """Test prediction under independence for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d+1, expected_shape)
    
    # Test prediction for all dimensions
    for axis in range(1, 5):
        pred_cat = cop.get_prediction_under_indep(axis)
        
        # Verify return type and value range
        assert isinstance(pred_cat, (int, np.int64))
        assert 1 <= pred_cat <= expected_shape[axis-1]
        
        # According to Proposition 2.1(c), we can validate against direct calculation
        cdf = cop.marginal_cdfs[axis-1][1:-1]  # Get breakpoints
        expected_cat = np.searchsorted(cdf, 0.5) + 1  # 1-indexed
        assert pred_cat == expected_cat
        
        # Verify prediction is consistent with manual calculation
        # For dimension 4 (last dimension)
        if axis == 4:
            assert pred_cat == 4

def test_4d_category_predictions_dataframe(cases_4d, expected_shape):
    """Test category predictions output format for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    var_names = {
        1: "First",
        2: "Second", 
        3: "Third",
        4: "Fourth"
    }
    
    predictors = [1,2,3]
    response = 4
    
    df = cop.get_predictions_ccr(
        predictors=predictors,
        response=response,
        variable_names=var_names
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    for axis in predictors:
        assert f"{var_names[axis]} Category" in df.columns
    assert f"Predicted {var_names[response]} Category" in df.columns
    
def test_plot_ccr_predictions_2d(table_4d):
    """Test 2D CCR prediction visualization."""
    # Setup
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    
    # Test both legend styles
    for style in ['side', 'xaxis']:
        # Call with 2D predictors
        fig = plt.figure()  # Create a figure to use
        assert fig is not None
        ccrvam.plot_ccr_predictions([1, 2], 4, legend_style=style)
        
        # Verify some figure was created
        assert plt.get_fignums()  # Check if any figures exist
        
        plt.close('all')

def test_plot_ccr_predictions_3d(table_4d):
    """Test 3D CCR prediction visualization with custom variable names."""
    # Setup
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    var_names = {1: "Length", 2: "Pain", 3: "Lordosis", 4: "Outcome"}
    
    # Test both legend styles
    for style in ['side', 'xaxis']:
        # Call with 3D predictors
        ccrvam.plot_ccr_predictions([1, 2, 3], 4, 
                                  variable_names=var_names,
                                  legend_style=style)
        
        # Verify some figure was created
        assert plt.get_fignums()  # Check if any figures exist
            
        plt.close('all')

def test_plot_ccr_predictions_custom_figsize(table_4d):
    """Test custom figsize for CCR prediction visualization."""
    # Setup
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    custom_figsize = (12, 10)
    
    # Call with custom figsize
    ccrvam.plot_ccr_predictions([1, 2], 4, figsize=custom_figsize)
    
    # Verify figure was created with correct size
    assert plt.get_fignums()
    fig = plt.gcf()  # Get current figure
    np.testing.assert_array_almost_equal(fig.get_size_inches(), custom_figsize)
    
    plt.close('all')

def test_plot_ccr_predictions_save_creates_directory(table_4d):
    """Test directory creation when saving plot."""
    # Setup
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a subdirectory path that doesn't exist yet
        subdir = os.path.join(tmpdirname, 'new_subdir')
        save_path = os.path.join(subdir, 'test_plot.png')
        
        # Test both legend styles
        for style in ['side', 'xaxis']:
            # Call with save_path
            ccrvam.plot_ccr_predictions([1, 2, 3], 4, 
                                      save_path=save_path,
                                      legend_style=style)
            
            # Assert directory was created and file exists
            assert os.path.exists(subdir)
            assert os.path.exists(save_path)
            
            # Check if separate legend file is created when needed
            legend_path = save_path.rsplit('.', 1)[0] + '_legend.png'
            if style == 'side' and os.path.exists(legend_path):
                assert os.path.exists(legend_path)
            
            plt.close('all')

def test_plot_ccr_predictions_invalid_predictors(table_4d):
    """Test error handling for invalid predictors."""
    # Setup
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    
    # Single predictor should work
    ccrvam.plot_ccr_predictions([1], 4)
    assert plt.get_fignums()  # Check if figure was created
    plt.close('all')
    
    # Empty list should raise error
    with pytest.raises(IndexError):
        ccrvam.plot_ccr_predictions([], 4)
        
    # Invalid predictor indices should raise error
    with pytest.raises(ValueError):
        ccrvam.plot_ccr_predictions([5], 4)
        
    plt.close('all')

def test_heatmap_content(table_4d):
    """Test that the plot contains expected visual elements."""
    # Setup 
    ccrvam = GenericCCRVAM.from_contingency_table(table_4d)
    
    # Create plot
    ccrvam.plot_ccr_predictions([1, 2], 4)
    
    # Get current figure
    fig = plt.gcf()
    ax = fig.axes[0]
    
    # Check y-axis has descending categories
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    assert y_labels == [str(i) for i in range(table_4d.shape[-1], 0, -1)]
    
    plt.close('all')
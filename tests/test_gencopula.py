import numpy as np
import pandas as pd
import pytest
from ccrvam import GenericCCRVAM

@pytest.fixture
def generic_copula():
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
    table[0,2,0,1] = 1
    table[0,2,0,4] = 2
    table[0,2,0,5] = 4
    
    # RDA Row 2 [0,2,1,*]
    table[0,2,1,3] = 1
    table[0,2,1,4] = 3
    
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
    table[0,0,0,4] = 2 
    table[0,0,0,5] = 2
    
    # RDA Row 6 [0,0,1,*]
    table[0,0,1,2] = 1
    table[0,0,1,3] = 1
    table[0,0,1,4] = 3
    
    # RDA Row 7 [1,2,0,*]
    table[1,2,0,2] = 3
    table[1,2,0,4] = 1
    table[1,2,0,5] = 2
    
    # RDA Row 8 [1,2,1,*]
    table[1,2,1,1] = 1
    table[1,2,1,4] = 3
    
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
    table[1,0,0,0] = 2
    table[1,0,0,1] = 2
    table[1,0,0,2] = 1
    table[1,0,0,3] = 5
    table[1,0,0,4] = 2
    
    # RDA Row 12 [1,0,1,*]
    table[1,0,1,0] = 2
    table[1,0,1,2] = 2
    table[1,0,1,3] = 3
    
    return table

@pytest.fixture
def cases_4d():
    """Fixture for 4D case-form data in 1-indexed format."""
    return np.array([
        # RDA Row 1
        [1,3,1,2],[1,3,1,5],[1,3,1,5],
        [1,3,1,6],[1,3,1,6],[1,3,1,6],[1,3,1,6],
        # RDA Row 2
        [1,3,2,4],[1,3,2,5],[1,3,2,5],[1,3,2,5],
        # RDA Row 3
        [1,2,1,2],[1,2,1,2],[1,2,1,3],[1,2,1,3],[1,2,1,3],
        [1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],[1,2,1,5],
        [1,2,1,6],[1,2,1,6],[1,2,1,6],[1,2,1,6],
        # RDA Row 4
        [1,2,2,2],[1,2,2,4],[1,2,2,4],[1,2,2,6],
        # RDA Row 5
        [1,1,1,5],[1,1,1,5],[1,1,1,6],[1,1,1,6],
        # RDA Row 6
        [1,1,2,3],[1,1,2,4],[1,1,2,5],[1,1,2,5],[1,1,2,5],
        # RDA Row 7
        [2,3,1,3],[2,3,1,3],[2,3,1,3],[2,3,1,5],[2,3,1,6],[2,3,1,6],
        # RDA Row 8
        [2,3,2,2],[2,3,2,5],[2,3,2,5],[2,3,2,5],
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
        [2,1,1,1],[2,1,1,1],[2,1,1,2],[2,1,1,2],[2,1,1,3],
        [2,1,1,4],[2,1,1,4],[2,1,1,4],[2,1,1,4],[2,1,1,4],
        [2,1,1,5],[2,1,1,5],
        # RDA Row 12
        [2,1,2,1],[2,1,2,1],[2,1,2,3],[2,1,2,3],
        [2,1,2,4],[2,1,2,4],[2,1,2,4]
    ])
    
@pytest.fixture
def expected_shape():
    """Fixture providing expected shape for the copula."""
    return (2, 3, 2, 6)

# Basic Creation Tests
def test_from_contingency_table_valid(contingency_table):
    """Test valid contingency table initialization."""
    copula = GenericCCRVAM.from_contingency_table(contingency_table)
    expected_P = contingency_table / contingency_table.sum()
    np.testing.assert_array_almost_equal(copula.P, expected_P)

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
def test_marginal_cdfs(generic_copula, expected_cdf_0, expected_cdf_1):
    """Test marginal CDF calculations."""
    np.testing.assert_almost_equal(generic_copula.marginal_cdfs[0], expected_cdf_0)
    np.testing.assert_almost_equal(generic_copula.marginal_cdfs[1], expected_cdf_1)

# Conditional PMF Tests
def test_conditional_pmfs(generic_copula):
    """Test conditional PMF calculations."""
    expected_1_given_0 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    pmf, _ = generic_copula._calculate_conditional_pmf(1, [0])
    np.testing.assert_array_almost_equal(pmf, expected_1_given_0)

# Regression Tests
@pytest.mark.parametrize("given_values, given_axes, target_axis, expected_value", [
    # Single conditioning
    ([0.0], [0], 1, 12/16),
    ([3/8], [0], 1, 6/16),
    ([1.0], [0], 1, 12/16),
])
def test_calculate_regression(generic_copula, given_values, given_axes, target_axis, expected_value):
    """Test regression calculation with multiple conditioning axes."""
    calculated = generic_copula._calculate_regression_batched(
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
def test_calculate_CCRAM(generic_copula, predictors, response, expected_ccram):
    """Test CCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM(predictors, response, scaled=False)
    np.testing.assert_almost_equal(calculated, expected_ccram)

# SCCRAM Tests
@pytest.mark.parametrize("predictors, response, expected_sccram", [
    ([1], 2, 0.84375/(12*0.0703125)),          # Single axis X1->X2
    ([2], 1, 0.0),                             # Single axis X2->X1
])
def test_calculate_SCCRAM(generic_copula, predictors, response, expected_sccram):
    """Test SCCRAM calculations with multiple conditioning axes."""
    calculated = generic_copula.calculate_CCRAM(predictors, response, scaled=True)
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
def test_predict_category_multi(generic_copula, source_categories, predictors, response, expected_categories):
    """Test category prediction with multiple conditioning axes."""
    predicted = generic_copula._predict_category_batched_multi(
        source_categories=source_categories,
        predictors=predictors,
        response=response
    )
    np.testing.assert_array_equal(predicted, expected_categories)

# Add Multi-axis Category Predictions Test
def test_get_predictions_ccr(generic_copula):
    """Test category predictions with multiple conditioning axes."""
    df = generic_copula.get_predictions_ccr(
        predictors=[2],
        response=1,
        axis_names={1: "Income", 2: "Education"}
    )
    
    assert isinstance(df, pd.DataFrame)
    assert "Predicted Income Category" in df.columns
    assert "Education Category" in df.columns

# Add Consistency Tests for Multi-axis
def test_multi_axis_consistency(generic_copula):
    """Test consistency between single and multiple axis calculations."""
    single_axis = generic_copula.calculate_CCRAM(1, 2)
    multi_axis = generic_copula.calculate_CCRAM([1], 2)
    np.testing.assert_almost_equal(single_axis, multi_axis)

# Invalid Cases Tests
def test_invalid_predictions(generic_copula):
    """Test invalid prediction handling."""
    with pytest.raises(IndexError):
        generic_copula._predict_category(5, 0, 1)

# Special Cases Tests
def test_prediction_special_cases(generic_copula):
    """Test edge cases in predictions."""
    single_pred = generic_copula._predict_category_batched_multi(np.array([0]), 0, 1)
    assert len(single_pred) == 1
    assert single_pred[0] == generic_copula._predict_category(0, 0, 1)

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
        
def test_calculate_ccs_valid(generic_copula):
    """Test valid calculation of scores."""
    scores_1 = generic_copula.calculate_ccs(1)
    scores_2 = generic_copula.calculate_ccs(2)

    # Check exact expected values
    expected_scores_1 = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875], dtype=np.float64)
    expected_scores_2 = np.array([0.125, 0.375, 0.75], dtype=np.float64)
    
    np.testing.assert_array_almost_equal(scores_1, expected_scores_1)
    np.testing.assert_array_almost_equal(scores_2, expected_scores_2)
    
def test_calculate_ccs_invalid_axis(generic_copula):
    """Test invalid axis handling for score calculation."""
    with pytest.raises(KeyError):
        generic_copula.calculate_ccs(3)  # Invalid axis index

def test_calculate_variance_ccs_valid(generic_copula):
    """Test valid calculation of score variance."""
    var_1 = generic_copula.calculate_variance_ccs(1)
    var_2 = generic_copula.calculate_variance_ccs(2)
    
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

def test_calculate_variance_ccs_invalid_axis(generic_copula):
    """Test invalid axis handling for variance calculation."""
    with pytest.raises(KeyError):
        generic_copula.calculate_variance_ccs(3)  # Invalid axis index

def test_from_cases_creation(cases_4d, table_4d, expected_shape):
    """Test creation of copula from cases data."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    assert cop.ndim == 4
    assert cop.P.shape == expected_shape
    assert np.all(cop.P >= 0)
    assert np.isclose(cop.P.sum(), 1.0)
    assert np.all(cop.contingency_table == table_4d)

def test_from_cases_marginal_pdfs(cases_4d, expected_shape):
    """Test marginal PDFs calculation from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test marginal PDFs exist for each dimension
    assert len(cop.marginal_pdfs) == 4
    
    # Test each marginal PDF sums to 1
    for axis in range(4):
        pdf = cop.marginal_pdfs[axis]
        assert np.isclose(np.sum(pdf), 1.0)

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

def test_from_cases_invalid_input():
    """Test error handling for invalid inputs."""
    invalid_cases = np.array([[0,1], [1,2]])  # Wrong number of dimensions
    invalid_shape = (2, 2)  # Wrong shape specification
    
    with pytest.raises(ValueError):
        GenericCCRVAM.from_cases(invalid_cases, (2,2,2,2))
    
    with pytest.raises(ValueError):
        GenericCCRVAM.from_cases(cases_4d, invalid_shape)

def test_from_cases_contingency_table(cases_4d, expected_shape):
    """Test contingency table properties from cases."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    # Test contingency table properties
    table = cop.contingency_table
    assert table.shape == expected_shape
    assert np.all(table >= 0)  # Non-negative counts
    assert np.sum(table) == len(cases_4d)  # Sum equals number of cases
    
def test_4d_ccram_calculations(cases_4d, expected_shape):
    """Test CCRAM calculations for 4D case with multiple conditioning axes."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
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
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
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

def test_4d_category_predictions_dataframe(cases_4d, expected_shape):
    """Test category predictions output format for 4D case."""
    cop = GenericCCRVAM.from_cases(cases_4d, expected_shape)
    
    axis_names = {
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
        axis_names=axis_names
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    for axis in predictors:
        assert f"{axis_names[axis]} Category" in df.columns
    assert f"Predicted {axis_names[response]} Category" in df.columns
import numpy as np
import pandas as pd
import pytest
from ccrvam.checkerboard.utils import (
    gen_contingency_to_case_form,
    gen_case_form_to_contingency
)
from ccrvam.checkerboard.utils import DataProcessor

@pytest.fixture
def contingency_table():
    """
    Fixture to create a sample contingency table.
    """
    return np.array([
        [0, 0, 20],
        [0, 10, 0],
        [20, 0, 0],
        [0, 10, 0],
        [0, 0, 20]
    ])
    
@pytest.fixture
def case_form_data():
    """
    Fixture to create a sample case-form data array.
    """
    return np.array([
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
        [3, 1], [3, 1], [3, 1], [3, 1], [3, 1],
        [3, 1], [3, 1], [3, 1], [3, 1], [3, 1],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
        [4, 2], [4, 2], [4, 2], [4, 2], [4, 2]
    ])
    
def test_gen_contingency_to_case_form(contingency_table, case_form_data):
    """
    Test gen_contingency_to_case_form conversion.
    """
    cases = gen_contingency_to_case_form(contingency_table)
    # Sort both arrays to ensure consistent comparison
    np.testing.assert_array_equal(
        cases[np.lexsort(cases.T)],
        case_form_data[np.lexsort(case_form_data.T)]
    )

def test_gen_case_form_to_contingency(contingency_table, case_form_data):
    """
    Test gen_case_form_to_contingency conversion.
    """
    reconstructed = gen_case_form_to_contingency(case_form_data, contingency_table.shape)
    np.testing.assert_array_equal(reconstructed, contingency_table)

@pytest.fixture
def gen_contingency_table():
    """Fixture for a simple 2D contingency table."""
    return np.array([
        [2, 1],
        [0, 3]
    ])

@pytest.fixture
def gen_case_form_data():
    """Fixture for corresponding case-form data."""
    return np.array([
        [0, 0], [0, 0],  # 2 cases
        [0, 1],          # 1 case
        [1, 1], [1, 1], [1, 1]  # 3 cases
    ])

@pytest.fixture
def gen_3d_cases():
    """Fixture for 3D batched case data."""
    return np.array([
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]]
    ])

def test_gen_contingency_to_case_form_2d(gen_contingency_table, gen_case_form_data):
    """Test gen_contingency_to_case_form conversion."""
    cases = gen_contingency_to_case_form(gen_contingency_table)
    # Sort both arrays to ensure consistent comparison
    np.testing.assert_array_equal(
        cases[np.lexsort(cases.T)],
        gen_case_form_data[np.lexsort(gen_case_form_data.T)]
    )

def test_gen_case_form_to_contingency_2d(gen_contingency_table, gen_case_form_data):
    """Test gen_case_form_to_contingency with 2D data."""
    reconstructed = gen_case_form_to_contingency(gen_case_form_data, gen_contingency_table.shape)
    np.testing.assert_array_equal(reconstructed, gen_contingency_table)

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

def test_case_form_to_contingency_nd_2d(contingency_table, case_form_data):
    """Test N-dimensional conversion with 2D data."""
    result = gen_case_form_to_contingency(case_form_data, contingency_table.shape)
    np.testing.assert_array_equal(result, contingency_table)

def test_contingency_to_case_form_nd_2d(contingency_table, case_form_data):
    """Test N-dimensional conversion with 2D data."""
    result = gen_contingency_to_case_form(contingency_table)
    # Sort both arrays for comparison
    np.testing.assert_array_equal(
        result[np.lexsort(result.T)],
        case_form_data[np.lexsort(case_form_data.T)]
    )
    
def test_case_form_to_contingency_nd_4d(table_4d, cases_4d):
    """Test N-dimensional conversion with 4D data."""
    result = gen_case_form_to_contingency(cases_4d, table_4d.shape)
    np.testing.assert_array_equal(result, table_4d)

def test_contingency_to_case_form_nd_4d(table_4d, cases_4d):
    """Test N-dimensional conversion with 4D data."""
    result = gen_contingency_to_case_form(table_4d)
    # Sort both arrays for comparison
    np.testing.assert_array_equal(
        result[np.lexsort(result.T)],
        cases_4d[np.lexsort(cases_4d.T)]
    )

@pytest.fixture
def sample_category_map():
    """Fixture for category mapping."""
    return {
        "pain": {
            "worse": 1,
            "same": 2,
            "slight.improvement": 3,
            "moderate.improvement": 4,
            "marked.improvement": 5,
            "complete.relief": 6
        }
    }

@pytest.fixture
def sample_var_list():
    """Fixture for variable names."""
    return ["x1", "x2", "x3", "pain"]

def test_load_case_form_array():
    """Test loading case form array."""
    data = np.array([[1, 2, 1, 5], [2, 1, 2, 3]])
    result = DataProcessor.load_data(
        data,
        data_form="case_form",
        dimension=(2, 3, 2, 6)
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 2, 6)

def test_load_freq_form_array():
    """Test loading frequency form array."""
    data = np.array([[1, 2, 1, 5, 2], [2, 1, 2, 3, 1]])
    result = DataProcessor.load_data(
        data,
        data_form="frequency_form",
        dimension=(2, 3, 2, 6)
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 2, 6)

def test_load_data_with_category_map(sample_category_map, sample_var_list):
    """Test loading data with category mapping."""
    df = pd.DataFrame({
        'x1': [1, 2],
        'x2': [2, 1],
        'x3': [1, 2],
        'pain': ['marked.improvement', 'moderate.improvement']
    })
    
    result = DataProcessor.load_data(
        df,
        data_form="case_form",
        dimension=(2, 3, 2, 6),
        var_list=sample_var_list,
        category_map=sample_category_map,
        named=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 2, 6)

def test_load_data_from_file(tmp_path, sample_category_map, sample_var_list):
    """Test loading data from file."""
    # Create temp file
    file_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'x1': [1, 2],
        'x2': [2, 1],
        'x3': [1, 2],
        'pain': ['marked.improvement', 'moderate.improvement']
    })
    df.to_csv(file_path, index=False)
    
    result = DataProcessor.load_data(
        str(file_path),
        data_form="case_form",
        dimension=(2, 3, 2, 6),
        var_list=sample_var_list,
        category_map=sample_category_map,
        named=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3, 2, 6)

def test_invalid_data_form():
    """Test invalid data form raises error."""
    with pytest.raises(ValueError):
        DataProcessor.load_data(
            np.array([[1, 2]]),
            data_form="invalid_form",
            dimension=(2, 2)
        )

def test_missing_file():
    """Test missing file raises error."""
    with pytest.raises(FileNotFoundError):
        DataProcessor.load_data(
            "nonexistent.csv",
            data_form="case_form",
            dimension=(2, 2)
        )

def test_invalid_category_mapping(sample_var_list):
    """Test invalid category mapping raises error."""
    df = pd.DataFrame({
        'x1': [1, 2],
        'x2': [2, 1],
        'pain': ['invalid', 'category']
    })
    
    with pytest.raises(IndexError):
        DataProcessor.load_data(
            df,
            data_form="case_form",
            dimension=(2, 2, 6),
            var_list=sample_var_list,
            category_map={'pain': {'valid': 1}},
            named=True
        )

def test_dimension_mismatch():
    """Test dimension mismatch raises error."""
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        DataProcessor.load_data(
            data,
            data_form="case_form",
            dimension=(2, 2, 2)  # Wrong dimension
        )

def test_data_type_conversion():
    """Test data type conversion."""
    df = pd.DataFrame({'x1': [1, 2], 'x2': [2, 1]})
    result = DataProcessor.load_data(
        df,
        data_form="case_form",
        dimension=(3, 3),
        named=True
    )
    assert result.dtype == np.int64

def test_load_caseform_data_with_delimiter(sample_var_list, sample_category_map, table_4d):
    """Test loading caseform data with delimiter."""
    table1 = DataProcessor.load_data(
        "./tests/data/caseform.pain.txt",
        data_form="case_form",
        dimension=(2, 3, 2, 6),
        var_list=sample_var_list,
        category_map=sample_category_map,
        named=True,
        delimiter="\t"
    )
    
    assert table_4d.shape == table1.shape
    assert table_4d.dtype == table1.dtype
    np.testing.assert_array_equal(table_4d, table1)
    
def test_load_freqform_data_with_delimiter(sample_var_list, sample_category_map, table_4d):
    """Test loading freqform data with delimiter."""
    table2 = DataProcessor.load_data(
        "./tests/data/freqform.pain.txt",
        data_form="frequency_form", 
        dimension=(2, 3, 2, 6),
        var_list=sample_var_list,
        category_map=sample_category_map,
        named=True,
        delimiter="\t"
    )
    
    assert table2.shape == table_4d.shape
    assert table2.dtype == table_4d.dtype
    np.testing.assert_array_equal(table2, table_4d)
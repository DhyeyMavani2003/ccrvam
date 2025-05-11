import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Optional
import warnings
class DataProcessor:
    """Data processing engine for contingency table analysis."""
    
    @staticmethod
    def load_data(
        data: Union[str, np.ndarray, pd.DataFrame],
        data_form: str,
        dimension: tuple,
        var_list: Optional[List[str]] = None,
        category_map: Optional[Dict[str, Dict[str, int]]] = None,
        named: bool = False,
        delimiter: str = None
    ) -> np.ndarray:
        """
        Load and process data for contingency table analysis.
        
        Input Arguments
        --------------
        - `data` : Data source - file path, raw data array, or data frame
            
        - `data_form` : Format of the data: "case_form", "frequency_form", or "table_form"

        - `dimension` : A tuple specifying the number of categories for each variable. 
                        The length of the tuple indicates the number of variables
                        , and each element in the tuple specifies the number of categories for the corresponding variable.
            
        - `var_list` : Names of variables in order of appearance in the data (optional)
            
        - `category_map` : Mapping of categorical labels to numeric indices for each variable (optional)
            
        - `named` : Whether the first row contains variable names (for file input)
            
        - `delimiter` : Column separator character for text files (optional)
            
        Outputs
        -------
        Processed contingency table for statistical analysis
            
        Warnings/Errors
        --------------
        - `ValueError` : If data_form is invalid or inputs are inconsistent
            
        - `FileNotFoundError` : If the specified data file cannot be found
        """
        # Validate inputs
        if data_form not in ["case_form", "frequency_form", "table_form"]:
            raise ValueError("data_form must be case_form, frequency_form, or table_form")

        # Handle file input
        if isinstance(data, str):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data}")
            
            if named:
                # Read with pandas to handle headers
                df = pd.read_csv(data_path, delimiter=delimiter)
                if var_list is None:
                    var_list = list(df.columns)
                data = df.to_numpy()
            else:
                # Read raw data
                data = np.loadtxt(data_path, delimiter=delimiter, skiprows=1 if named else 0)
                
        elif isinstance(data, pd.DataFrame):
            if var_list is None and named:
                var_list = list(data.columns)
            data = data.to_numpy()

        # Convert string categories to numeric if needed
        if category_map is not None:
            data = DataProcessor._apply_category_mapping(
                data, 
                category_map,
                var_list,
                data_form
            )

        # Process based on data form
        if data_form == "case_form":
            return DataProcessor._process_case_form(data, dimension)
        elif data_form == "frequency_form":
            return DataProcessor._process_frequency_form(data, dimension)
        else:
            return DataProcessor._process_table_form(data, dimension)

    @staticmethod
    def _apply_category_mapping(
        data: np.ndarray,
        category_map: Dict[str, Dict[str, int]],
        var_list: List[str],
        data_form: str
    ) -> np.ndarray:
        """Internal helper to convert qualitative categories to numerical categories (1, 2, ...)."""
        if data.dtype.kind in 'ifu':  # Already numeric
            return data
            
        result = data.copy()
        
        def _map_category(value, var_map):
            """Internal helper to map categories with error handling"""
            if isinstance(value, str):
                value = value.strip()  # Remove whitespace
                return var_map.get(value, value)
            return value
        
        if data_form in ["case_form", "frequency_form"]:
            # Handle variables with category mappings
            n_cols = data.shape[1]
            n_vars = len(var_list) if var_list else n_cols
            
            for i in range(n_vars):
                var = var_list[i] if var_list else str(i)
                if var in category_map:
                    try:
                        result[:, i] = [_map_category(x, category_map[var]) for x in data[:, i]]
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Error converting categories for variable '{var}': {e}")
        
        # Convert to numeric, handling any remaining string values
        try:
            result = result.astype(float)
        except ValueError as e:
            # Find problematic values
            mask = np.array([[not isinstance(x, (int, float)) for x in row] for row in result])
            if np.any(mask):
                bad_values = result[mask]
                raise ValueError(f"Could not convert categories to numeric: {set(bad_values)}")
            raise e
            
        return result

    @staticmethod
    def _process_case_form(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Internal helper to convert case form data to contingency table."""
        if data.ndim != 2:
            raise ValueError("Case form data must be 2D array")
            
        if data.shape[1] != len(shape):
            raise ValueError("Number of variables doesn't match shape")
            
        # Adjust for 0-based indexing
        data = data - 1
            
        # Initialize contingency table
        table = np.zeros(shape, dtype=int)
        
        # Count occurrences
        for case in data:
            idx = tuple(int(i) for i in case)
            table[idx] += 1
            
        return table

    @staticmethod 
    def _process_frequency_form(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Internal helper to convert frequency form data to contingency table."""
        if data.ndim != 2:
            raise ValueError("Frequency form data must be 2D array")
            
        if data.shape[1] != len(shape) + 1:
            raise ValueError("Frequency form should have n+1 columns (n variables + frequency)")
            
        # Split into cases and frequencies
        cases = data[:, :-1] - 1  # Adjust for 0-based indexing
        freqs = data[:, -1]
        
        # Initialize contingency table
        table = np.zeros(shape, dtype=int)
        
        # Fill table with frequencies
        for case, freq in zip(cases, freqs):
            idx = tuple(int(i) for i in case)
            table[idx] = int(freq)
            
        return table

    @staticmethod
    def _process_table_form(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Internal helper to process table form data."""
        if data.shape != shape:
            raise ValueError(f"Table shape {data.shape} doesn't match specified shape {shape}")
        
        # Log a warning if the data is not an integer, but not an error
        if not np.issubdtype(data.dtype, np.integer):
            warnings.warn("[WARNING]Table form data has non-integer values", UserWarning)
        
        return data
        
def gen_contingency_to_case_form(
    contingency_table: np.ndarray
) -> np.ndarray:
    """
    Convert a multi-dimensional contingency table data to the case form data.
    
    Input Arguments
    --------------
    - `contingency_table` : Multi-dimensional contingency table containing frequency counts

    Outputs
    -------
    Array for the case form data frames containing individual observations, with one or more categorical variables
    """
    # Get indices of non-zero elements
    indices = np.nonzero(contingency_table)
    counts = contingency_table[indices]
    
    # Create cases list
    cases = []
    for idx, count in zip(zip(*indices), counts):
        cases.extend([list(idx)] * int(count))
    
    return np.array(cases)

def gen_case_form_to_contingency(
    cases: np.ndarray,
    shape: tuple,
    axis_order: Optional[list] = None
) -> np.ndarray:
    """
    Convert case form data to a multi-dimensional contingency table.
    
    Input Arguments
    --------------
    - `cases` : Array where each row represents an observation with categorical variables

    - `shape` : Dimensions of the output contingency table

    - `axis_order` : (Optional) List specifying how case columns map to contingency table dimensions. 
                     For example, if cases has columns [A,B,C] and axis_order is [2,0,1],
                     then column A maps to dimension 2, B to 0, and C to 1 in the contingency table.
                     If None, assumes sequential mapping [0,1,2,...].
    
    Outputs
    -------
    Multi-dimensional contingency table.
    """
    if axis_order is None:
        axis_order = list(range(cases.shape[1]))
        
    table = np.zeros(shape, dtype=int)
    n_axes = len(shape)
    
    # Create full index with zeros for missing axes
    def _get_full_index(case, axis_order):
        """Internal helper to create full index from case."""
        idx = [0] * n_axes
        for i, axis in enumerate(axis_order):
            idx[axis] = int(case[i])
        return tuple(idx)
    
    # Handle both 2D and 3D cases
    if cases.ndim == 3:
        # For batched data
        for batch in cases:
            for case in batch:
                idx = _get_full_index(case, axis_order)
                table[idx] += 1
    else:
        # For single batch
        for case in cases:
            idx = _get_full_index(case, axis_order)
            table[idx] += 1
            
    return table
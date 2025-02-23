import numpy as np
        
def gen_contingency_to_case_form(contingency_table: np.ndarray) -> np.ndarray:
    """Convert N-dimensional contingency table to case form.
    
    Parameters
    ----------
    contingency_table : np.ndarray
        N-dimensional contingency table
        
    Returns
    -------
    np.ndarray
        Array of cases where each row represents coordinates
    """
    # Get indices of non-zero elements
    indices = np.nonzero(contingency_table)
    counts = contingency_table[indices]
    
    # Create cases list
    cases = []
    for idx, count in zip(zip(*indices), counts):
        cases.extend([list(idx)] * int(count))
    
    return np.array(cases)

def gen_case_form_to_contingency(cases: np.ndarray, 
                                shape: tuple,
                                axis_order: list = None) -> np.ndarray:
    """Convert cases to contingency table with specified axis ordering.
    
    Parameters
    ----------
    cases : np.ndarray
        Array of cases where each row is a sample
    shape : tuple
        Shape of output contingency table
    axis_order : list, optional
        Order of axes for reconstruction
    
    Returns
    -------
    np.ndarray
        Reconstructed contingency table
    """
    if axis_order is None:
        axis_order = list(range(cases.shape[1]))
        
    table = np.zeros(shape, dtype=int)
    n_axes = len(shape)
    
    # Create full index with zeros for missing axes
    def get_full_index(case, axis_order):
        idx = [0] * n_axes
        for i, axis in enumerate(axis_order):
            idx[axis] = int(case[i])
        return tuple(idx)
    
    # Handle both 2D and 3D cases
    if cases.ndim == 3:
        # For batched data
        for batch in cases:
            for case in batch:
                idx = get_full_index(case, axis_order)
                table[idx] += 1
    else:
        # For single batch
        for case in cases:
            idx = get_full_index(case, axis_order)
            table[idx] += 1
            
    return table
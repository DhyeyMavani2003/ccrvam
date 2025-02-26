import numpy as np
from scipy.stats import bootstrap, permutation_test
from dataclasses import dataclass
from typing import Union, Tuple, List
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from .gencopula import GenericCCRVAM
from .utils import gen_contingency_to_case_form, gen_case_form_to_contingency

@dataclass
class CustomBootstrapResult:
    """Container for bootstrap simulation results with visualization capabilities.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric being bootstrapped
    observed_value : float
        Original observed value of the metric
    confidence_interval : Tuple[float, float]
        Lower and upper confidence interval bounds
    bootstrap_distribution : np.ndarray
        Array of bootstrapped values
    standard_error : float 
        Standard error of the bootstrap distribution
    histogram_fig : plt.Figure, optional
        Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    standard_error: float
    histogram_fig: plt.Figure = None

    def plot_distribution(self, title=None):
            """Plot bootstrap distribution with observed value."""
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                data_range = np.ptp(self.bootstrap_distribution)
                bins = 1 if data_range == 0 else min(50, max(1, int(np.sqrt(len(self.bootstrap_distribution)))))
                
                ax.hist(self.bootstrap_distribution, bins=bins, density=True, alpha=0.7)
                ax.axvline(self.observed_value, color='red', linestyle='--', 
                        label=f'Observed {self.metric_name}')
                ax.set_xlabel(f'{self.metric_name} Value')
                ax.set_ylabel('Density')
                ax.set_title(title or 'Bootstrap Distribution')
                ax.legend()
                self.histogram_fig = fig
                return fig
            except Exception as e:
                print(f"Warning: Could not create plot: {str(e)}")
                return None

def bootstrap_ccram(contingency_table: np.ndarray,
                   predictors: Union[List[int], int],
                   response: int, 
                   scaled: bool = False,
                   n_resamples: int = 9999,
                   confidence_level: float = 0.95,
                   method: str = 'percentile',
                   random_state = None) -> CustomBootstrapResult:
    """Perform bootstrap simulation for (S)CCRAM measure.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        Input contingency table
    predictors : Union[List[int], int]
        List of 1-indexed predictors axes for category prediction
    response : int
        1-indexed target response axis for category prediction
    scaled : bool, default=False
        Whether to use scaled CCRAM (SCCRAM)
    n_resamples : int, default=9999
        Number of bootstrap resamples
    confidence_level : float, default=0.95
        Confidence level for intervals
    method : str, default='percentile'
        Bootstrap CI method
    random_state : optional
        Random state for reproducibility
        
    Returns
    -------
    CustomBootstrapResult
        Bootstrap results including CIs and distribution
    """
    if not isinstance(predictors, (list, tuple)):
        predictors = [predictors]
        
    # Input validation and 0-indexing
    parsed_predictors = []
    for pred_axis in predictors:
        parsed_predictors.append(pred_axis - 1)
    parsed_response = response - 1
    
    # Validate dimensions
    ndim = contingency_table.ndim
    if parsed_response >= ndim:
        raise ValueError(f"Response axis {response} is out of bounds for array of dimension {ndim}")
    
    for axis in parsed_predictors:
        if axis >= ndim:
            raise ValueError(f"Predictor axis {axis+1} is out of bounds for array of dimension {ndim}")

    # Format metric name
    predictors_str = ",".join(f"X{i}" for i in predictors)
    metric_name = f"{'SCCRAM' if scaled else 'CCRAM'} ({predictors_str}) to X{response}"
    
    # Calculate observed value
    gen_copula = GenericCCRVAM.from_contingency_table(contingency_table)
    observed_ccram = gen_copula.calculate_CCRAM(predictors, response, scaled)
    
    # Get all required axes in sorted order
    all_axes = sorted(parsed_predictors + [parsed_response])
    
    # Create full axis order including unused axes
    # full_axis_order = all_axes + [i for i in range(ndim) if i not in all_axes]
    
    # Convert to case form using complete axis order
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Split variables based on position in all_axes
    axis_positions = {axis: i for i, axis in enumerate(all_axes)}
    source_data = [cases[:, axis_positions[axis]] for axis in parsed_predictors]
    target_data = cases[:, axis_positions[parsed_response]]
    data = (*source_data, target_data)

    def ccram_stat(*args, axis=0):
        if args[0].ndim > 1:
            batch_size = args[0].shape[0]
            source_data = args[:-1]
            target_data = args[-1]
            
            cases = np.stack([
                np.column_stack([source[i].reshape(-1, 1) for source in source_data] + 
                              [target_data[i].reshape(-1, 1)])
                for i in range(batch_size)
            ])
        else:
            cases = np.column_stack([arg.reshape(-1, 1) for arg in args])
            
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=all_axes
                )
                copula = GenericCCRVAM.from_contingency_table(table)
                value = copula.calculate_CCRAM(predictors, response, scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=all_axes
            )
            copula = GenericCCRVAM.from_contingency_table(table)
            return copula.calculate_CCRAM(predictors, response, scaled)

    res = bootstrap(
        data,
        ccram_stat,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )
    
    result = CustomBootstrapResult(
        metric_name=metric_name,
        observed_value=observed_ccram,
        confidence_interval=res.confidence_interval,
        bootstrap_distribution=res.bootstrap_distribution,
        standard_error=res.standard_error
    )
    
    result.plot_distribution(f'Bootstrap Distribution: {metric_name}')
    return result

def _bootstrap_predict_category_multi(
    contingency_table: np.ndarray,
    source_categories: List[int],
    predictors: List[int],
    response: int,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state = None
):
    """Bootstrap confidence intervals for multi-axis category prediction."""
    # Get all required axes in sorted order
    all_axes = sorted(predictors + [response])
    
    # Convert table to case form
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Create axis mapping
    axis_positions = {axis: i for i, axis in enumerate(all_axes)}
    
    # Split variables using axis mapping
    source_data = [cases[:, axis_positions[axis]] for axis in predictors]
    target_data = cases[:, axis_positions[response]]
    data = (*source_data, target_data)

    def prediction_stat(*args, axis=0):
        source_data = args[:-1]
        target_data = args[-1]
        
        if source_data[0].ndim > 1:
            batch_size = source_data[0].shape[0]
            cases = np.stack([
                np.column_stack([
                    source[i].reshape(-1, 1) for source in source_data
                ] + [target_data[i].reshape(-1, 1)])
                for i in range(batch_size)
            ])
        else:
            cases = np.column_stack([
                arg.reshape(-1, 1) for arg in args
            ])
            
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases,
                    shape=contingency_table.shape,
                    axis_order=all_axes  # Pass correct axis order
                )
                copula = GenericCCRVAM.from_contingency_table(table)
                pred = copula._predict_category_batched_multi(
                    source_categories, predictors, response
                )
                results.append(pred)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=all_axes  # Pass correct axis order
            )
            copula = GenericCCRVAM.from_contingency_table(table)
            return copula._predict_category_batched_multi(
                source_categories, predictors, response
            )

    return bootstrap(
        data,
        prediction_stat,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state,
        paired=True,
        vectorized=True
    )

def bootstrap_predict_category_summary(
    contingency_table: np.ndarray,
    predictors: List[int],
    predictors_names: List[str],
    response: int,
    response_name: str = "Y",
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state = None
) -> pd.DataFrame:
    """Generate bootstrap summary table for multi-axis predictions.
    
    Parameters
    ----------
    contingency_table : np.ndarray
        Contingency table
    predictors : List[int]
        List of 1-indexed predictors axes for category prediction
    predictors_names : List[str]
        Source axes names
    response : int
        1-indexed target response axis for category prediction
    response_name : str, default='Y'
        Target axis name
    n_resamples : int, default=9999
        Number of resamples
    confidence_level : float, default=0.95
        Confidence level
    method : str, default='percentile'
        Bootstrap CI method
    random_state : optional
        Random state

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame of prediction summary
    """
    # Input validation and 0-indexing
    parsed_predictors = []
    for pred_axis in predictors:
        parsed_predictors.append(pred_axis - 1)
    parsed_response = response - 1
    
    # Validate dimensions
    ndim = contingency_table.ndim
    if parsed_response >= ndim:
        raise ValueError(f"Response axis {response} is out of bounds")
    for axis in parsed_predictors:
        if axis >= ndim:
            raise ValueError(f"Predictor axis {axis+1} is out of bounds")
    
    # Get all required axes in sorted order
    # all_axes = sorted(parsed_predictors + [parsed_response])
    
    # Get dimensions for each source axis and target axis
    source_dims = [contingency_table.shape[axis] for axis in parsed_predictors]
    target_dim = contingency_table.shape[parsed_response]
    
    # Create all combinations of source categories
    source_categories = np.array(np.meshgrid(
        *[range(dim) for dim in source_dims]
    )).T.reshape(-1, len(parsed_predictors))
    
    results = []
    for cats in source_categories:
        res = _bootstrap_predict_category_multi(
            contingency_table,
            cats.tolist(),
            parsed_predictors,
            parsed_response,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
            random_state=random_state
        )
        results.append(res)
    
    # Initialize multi-dimensional summary array
    summary_shape = tuple([target_dim] + source_dims)
    summary = np.zeros(summary_shape)
    
    # Fill summary array
    for idx, res in enumerate(results):
        source_indices = np.unravel_index(idx, source_dims)
        bootstrap_preds = res.bootstrap_distribution
        unique_preds, counts = np.unique(bootstrap_preds, return_counts=True)
        total = len(bootstrap_preds)
        
        for val, count in zip(unique_preds, counts):
            summary[(int(val),) + source_indices] = (count / total) * 100
            
    # Create multi-index for source categories
    source_names = [
        [f"{name}={i}" for i in range(dim)]
        for name, dim in zip(predictors_names, source_dims)
    ]
    
    # Create target categories
    target_categories = [f"{response_name}={i}" for i in range(summary.shape[0])]
    
    # Reshape summary matrix for DataFrame
    reshaped_summary = summary.reshape(summary.shape[0], -1)
    
    # Create multi-index columns
    column_tuples = list(itertools.product(*source_names))
    columns = pd.MultiIndex.from_tuples(column_tuples)
    
    # Create DataFrame
    summary_df = pd.DataFrame(
        reshaped_summary,
        index=target_categories,
        columns=columns
    )
    
    print("\nPrediction Summary (% of bootstrap samples)")
    print("-" * 80)
    print(summary_df.round(1).to_string(float_format=lambda x: f"{x:5.1f}%"))
    print("-" * 80)
    
    return summary_df

@dataclass 
class CustomPermutationResult:
    """Container for permutation test results with visualization capabilities.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric being tested
    observed_value : float
        Original observed value
    p_value : float
        Permutation test p-value
    null_distribution : np.ndarray
        Array of values under null hypothesis
    histogram_fig : plt.Figure, optional
        Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    p_value: float
    null_distribution: np.ndarray
    histogram_fig: plt.Figure = None

    def plot_distribution(self, title=None):
        """Plot null distribution with observed value."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            data_range = np.ptp(self.null_distribution)
            bins = 1 if data_range == 0 else min(50, max(1, int(np.sqrt(len(self.null_distribution)))))
            
            ax.hist(self.null_distribution, bins=bins, density=True, alpha=0.7)
            ax.axvline(self.observed_value, color='red', linestyle='--', 
                      label=f'Observed {self.metric_name}')
            ax.set_xlabel(f'{self.metric_name} Value')
            ax.set_ylabel('Density')
            ax.set_title(title or 'Null Distribution')
            ax.legend()
            self.histogram_fig = fig
            return fig
        except Exception as e:
            print(f"Warning: Could not create plot: {str(e)}")
            return None

def permutation_test_ccram(contingency_table: np.ndarray,
                          predictors: Union[List[int], int],
                          response: int,
                          scaled: bool = False,
                          alternative: str = 'greater',
                          n_resamples: int = 9999,
                          random_state = None) -> CustomPermutationResult:
    """Perform permutation test for (S)CCRAM measure.
    
    Parameters
    ----------
    contingency_table : numpy.ndarray
        Input contingency table
    predictors : Union[List[int], int]
        List of 1-indexed predictors axes for category prediction
    response : int
        1-indexed target response axis for category prediction
    scaled : bool, default=False
        Whether to use scaled CCRAM (SCCRAM)
    alternative : str, default='greater'
        Alternative hypothesis ('greater', 'less', 'two-sided')
    n_resamples : int, default=9999
        Number of permutations
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    CustomPermutationResult
        Test results including p-value and null distribution
    """
    if not isinstance(predictors, (list, tuple)):
        predictors = [predictors]
        
    # Input validation and 0-indexing
    parsed_predictors = []
    for pred_axis in predictors:
        parsed_predictors.append(pred_axis - 1)
    parsed_response = response - 1
    
    # Validate dimensions
    ndim = contingency_table.ndim
    if parsed_response >= ndim:
        raise ValueError(f"Response axis {response} is out of bounds for array of dimension {ndim}")
    
    for axis in parsed_predictors:
        if axis >= ndim:
            raise ValueError(f"Predictor axis {axis+1} is out of bounds for array of dimension {ndim}")

    # Format metric name
    predictors_str = ",".join(f"X{i}" for i in predictors)
    metric_name = f"{'SCCRAM' if scaled else 'CCRAM'} ({predictors_str}) to X{response}"
    
    # Get all required axes in sorted order
    all_axes = sorted(parsed_predictors + [parsed_response])
    
    # Create full axis order including unused axes
    # full_axis_order = all_axes + [i for i in range(ndim) if i not in all_axes]
    
    # Convert to case form using complete axis order
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Split variables based on position in all_axes
    axis_positions = {axis: i for i, axis in enumerate(all_axes)}
    source_data = [cases[:, axis_positions[axis]] for axis in parsed_predictors]
    target_data = cases[:, axis_positions[parsed_response]]
    data = (*source_data, target_data)

    def ccram_stat(*args, axis=0):
        if args[0].ndim > 1:
            batch_size = args[0].shape[0]
            source_data = args[:-1]
            target_data = args[-1]
            
            cases = np.stack([
                np.column_stack([source[i].reshape(-1, 1) for source in source_data] + 
                              [target_data[i].reshape(-1, 1)])
                for i in range(batch_size)
            ])
        else:
            cases = np.column_stack([arg.reshape(-1, 1) for arg in args])
            
        if cases.ndim == 3:
            results = []
            for batch_cases in cases:
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=all_axes
                )
                copula = GenericCCRVAM.from_contingency_table(table)
                value = copula.calculate_CCRAM(predictors, response, scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=all_axes
            )
            copula = GenericCCRVAM.from_contingency_table(table)
            return copula.calculate_CCRAM(predictors, response, scaled)

    perm = permutation_test(
        data,
        ccram_stat,
        permutation_type='pairings',
        alternative=alternative,
        n_resamples=n_resamples,
        random_state=random_state,
        vectorized=True
    )
    
    result = CustomPermutationResult(
        metric_name=metric_name,
        observed_value=perm.statistic,
        p_value=perm.pvalue,
        null_distribution=perm.null_distribution
    )
    
    result.plot_distribution(f'Null Distribution: {metric_name}')
    return result
import numpy as np
from scipy.stats import bootstrap, permutation_test
from dataclasses import dataclass
from typing import Union, Tuple, List
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from .genccrvam import GenericCCRVAM
from .utils import gen_contingency_to_case_form, gen_case_form_to_contingency

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class CustomBootstrapResult:
    """Container for bootstrap simulation results with visualization capabilities.
    
    Parameters
    ----------
    - metric_name : `str`
        Name of the metric being bootstrapped
    - observed_value : `float`
        Original observed value of the metric
    - confidence_interval : `Tuple[float, float]`
        Lower and upper confidence interval bounds
    - bootstrap_distribution : `np.ndarray`
        Array of bootstrapped values
    - standard_error : `float` 
        Standard error of the bootstrap distribution
    - bootstrap_tables : `np.ndarray`, optional
        Array of bootstrapped contingency tables
    - histogram_fig : `plt.Figure`, optional
        Matplotlib figure of distribution plot
    """
    
    metric_name: str 
    observed_value: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    standard_error: float
    bootstrap_tables: np.ndarray = None
    histogram_fig: plt.Figure = None

    def plot_distribution(self, title=None):
        """Plot bootstrap distribution with observed value."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_range = np.ptp(self.bootstrap_distribution)
            
            # Handle both exact zeros and very small ranges due to floating-point precision
            if data_range < 1e-10:  # Choose an appropriate small threshold
                # Almost degenerate case - all values are approximately the same
                unique_value = np.mean(self.bootstrap_distribution)
                ax.axvline(unique_value, color='blue', linewidth=2, 
                        label=f'All bootstrap values ≈ {unique_value:.4f}')
                ax.set_xlim([unique_value - 0.1, unique_value + 0.1])  # Add some padding
                ax.text(unique_value, 0.5, f"All {len(self.bootstrap_distribution)} bootstrap\nvalues ≈ {unique_value:.4f}", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            else:
                # Normal case - use histogram
                bins = min(50, max(10, int(np.sqrt(len(self.bootstrap_distribution)))))
                ax.hist(self.bootstrap_distribution, bins=bins, density=True, alpha=0.7)
            
            # Always show observed value
            ax.axvline(self.observed_value, color='red', linestyle='--', 
                    label=f'Observed {self.metric_name} = {self.observed_value:.4f}')
            
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
                   random_state = None,
                   store_tables: bool = False) -> CustomBootstrapResult:
    """Perform bootstrap simulation for (S)CCRAM measure.
    
    Parameters
    ----------
    - contingency_table : `np.ndarray`
        Input contingency table
    - predictors : `Union[List[int], int]`
        List of 1-indexed predictors axes for category prediction
    - response : `int`
        1-indexed target response axis for category prediction
    - scaled : `bool`, default=False
        Whether to use scaled CCRAM (SCCRAM)
    - n_resamples : `int`, default=9999
        Number of bootstrap resamples
    - confidence_level : `float`, default=0.95
        Confidence level for intervals
    - method : `str`, default='percentile'
        Bootstrap CI method
    - random_state : `int`, optional
        Random state for reproducibility
    - store_tables : `bool`, default=False
        Whether to store the bootstrapped contingency tables
        
    Returns
    -------
    - `CustomBootstrapResult`
        Bootstrap results including CIs, distribution and tables
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
    gen_ccrvam = GenericCCRVAM.from_contingency_table(contingency_table)
    observed_ccram = gen_ccrvam.calculate_CCRAM(predictors, response, scaled)
    
    # Get all required axes in sorted order
    all_axes = sorted(parsed_predictors + [parsed_response])
    
    # Create full axis order including unused axes
    # full_axis_order = all_axes + [i for i in range(ndim) if i not in all_axes]
    
    # Convert to case form using complete axis order
    cases = gen_contingency_to_case_form(contingency_table)
    
    # Store bootstrap tables if requested
    bootstrap_tables = None
    if store_tables:
        bootstrap_tables = np.zeros((n_resamples,) + contingency_table.shape)
    
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
            for i, batch_cases in enumerate(cases):
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=all_axes
                )
                
                # Store table if requested
                if store_tables and bootstrap_tables is not None and i < n_resamples:
                    bootstrap_tables[i] = table
                
                ccrvam = GenericCCRVAM.from_contingency_table(table)
                value = ccrvam.calculate_CCRAM(predictors, response, scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=all_axes
            )
            ccrvam = GenericCCRVAM.from_contingency_table(table)
            return ccrvam.calculate_CCRAM(predictors, response, scaled)

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
        standard_error=res.standard_error,
        bootstrap_tables=bootstrap_tables
    )
    
    result.plot_distribution(f'Bootstrap Distribution: {metric_name}')
    return result

def bootstrap_predict_ccr_summary(
    table,
    predictors,
    predictors_names=None,
    response=None,
    response_name=None,
    n_resamples=9999,
    random_state=None
):
    """
    Compute bootstrap prediction matrix with percentages for CCR.
    
    Parameters:
    -----------
    - table : `np.ndarray`
        Contingency table. Can be 2D, 3D, 4D, etc.
    - predictors : `List`
        List of predictor dimensions (1-indexed).
    - predictors_names : `List`, optional
        Names for the predictor dimensions.
    - response : `int`, optional
        Index of the response dimension (1-indexed). If None, the last dimension is used.
    - response_name : `str`, optional
        Name for the response dimension.
    - n_resamples : `int`, optional
        Number of bootstrap resamples.
    - random_state : `int`, optional
        Random seed for reproducibility.
    
    Returns:
    --------
    - `pd.DataFrame`
        CCR Prediction matrix post-bootstrap showing percentage for each combination of predictor values.
    """
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Determine response dimension if not specified
    if response is None:
        response = table.ndim
    else:
        # Convert 1-indexed to 0-indexed
        response = response - 1
    
    # Convert predictors from 1-indexed to 0-indexed
    parsed_predictors = [p - 1 for p in predictors]
    
    # Generate default names if not provided
    if predictors_names is None:
        predictors_names = [f"X{i+1}" for i in predictors]
    if response_name is None:
        response_name = f"Y = X{response+1}"
    
    # Get dimensions for each axis
    dims = table.shape
    pred_dims = [dims[p] for p in parsed_predictors]
    response_dim = dims[response]
    
    # Convert table to case form for resampling
    cases = gen_contingency_to_case_form(table)
    
    # Create all possible combinations of predictor values (1-indexed for output)
    pred_combinations = list(itertools.product(*[range(1, dim+1) for dim in pred_dims]))
    
    # Create column headers
    columns = []
    for combo in pred_combinations:
        header = " ".join([f"{name}={val}" for name, val in zip(predictors_names, combo)])
        columns.append(header)
    
    # Create row labels (1-indexed for output)
    rows = [f"{response_name}={i+1}" for i in range(response_dim)]
    
    # Initialize result matrix with zeros
    result = np.zeros((response_dim, len(pred_combinations)))
    
    # For each bootstrap sample
    for _ in range(n_resamples):
        # Generate bootstrap sample
        bootstrap_indices = np.random.choice(len(cases), len(cases), replace=True)
        bootstrap_cases = cases[bootstrap_indices]
        
        # Generate contingency table from bootstrap cases
        bootstrap_table = gen_case_form_to_contingency(bootstrap_cases, shape=dims)
        
        # Create CCRVAM model for this bootstrap sample
        ccrvam = GenericCCRVAM.from_contingency_table(bootstrap_table)
        
        # For each predictor combination, get the predicted category
        for i, combo in enumerate(pred_combinations):
            # Convert to 0-indexed for internal processing
            source_cats = [c-1 for c in combo]
            
            # Predict category (returns 0-indexed result)
            try:
                predicted = ccrvam._predict_category(
                    source_category=source_cats,
                    predictors=parsed_predictors,
                    response=response
                )
                
                # Increment count for this prediction
                result[predicted, i] += 1
            except Exception:
                # Skip if prediction fails (e.g., due to zero probabilities)
                continue
    
    # Convert counts to percentages
    col_sums = result.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentages = np.where(col_sums > 0, 
                              (result / col_sums[:, np.newaxis].T) * 100, 
                              0)
    
    # Create DataFrame
    summary_df = pd.DataFrame(percentages, index=rows, columns=columns)
    
    # Remove rows with all zeros
    summary_df = summary_df.loc[(summary_df != 0).any(axis=1)]
    
    # Create DataFrame with Predicted Categories using get_predictions_ccr but in the same format as summary_df
    
    # Initialize CCRVAM model on original table
    ccrvam_orig = GenericCCRVAM.from_contingency_table(table)
    
    # Create variable names mapping for get_predictions_ccr (1-indexed)
    var_names = {}
    for i, name in enumerate(predictors_names):
        var_names[predictors[i]] = name
    
    # Add response name to variable names
    var_names[response + 1] = response_name.replace("Y = ", "")
    
    # Get predictions from original table
    predictions_df = ccrvam_orig.get_predictions_ccr(predictors, response + 1, var_names)
    
    # Create a simplified DataFrame for predictions with just one row
    # Instead of the matrix format, this shows the predicted category for each combination
    pred_df = pd.DataFrame(index=["Predicted"], columns=columns)
    
    for _, row in predictions_df.iterrows():
        # Extract predictor categories and format them like summary_df column names
        pred_values = []
        for i, p in enumerate(predictors):
            col_name = f"{predictors_names[i]} Category"
            pred_values.append(f"{predictors_names[i]}={int(row[col_name])}")
        
        # Create the column name in the same format as summary_df
        col_name = " ".join(pred_values)
        
        # Get the predicted category (1-indexed)
        response_col = [c for c in predictions_df.columns if "Predicted" in c][0]
        pred_cat = int(row[response_col])
        
        # Store the category number directly
        pred_df.loc["Predicted", col_name] = pred_cat
    
    # Add predictions DataFrame as an attribute to the summary DataFrame
    summary_df.predictions = pred_df
    
    def plot_prediction_heatmap(df=summary_df, figsize=None, cmap='Blues', 
                            show_values=True, save_path=None, dpi=300):
        """
        Plot prediction percentages as a heatmap visualization.
        
        Parameters
        ----------
        - figsize : `Tuple`, optional
            Figure size (width, height)
        - cmap : `str`, optional
            Colormap for heatmap
        - show_values : `bool`, optional
            Whether to show percentage values
        - save_path : `str`, optional
            Path to save the plot
        - dpi : `int`, optional
            Resolution for saved image
        """
        # Get data dimensions
        n_rows, n_cols = df.shape
        
        # Set figure size based on data dimensions
        if figsize is None:
            figsize = (max(8, min(n_cols * 1.2, 16)), 
                    max(6, min(n_rows * 1.2, 12)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort the DataFrame by index in descending order to get categories from highest to lowest
        df_sorted = df.sort_index(ascending=False)
        
        # Create heatmap with sorted data
        im = ax.imshow(df_sorted.values, cmap=cmap, aspect='auto')
        
        # Check if we have predictions attribute
        has_predictions = hasattr(df, 'predictions')
        
        # Add text values if requested
        if show_values:
            for i in range(n_rows):
                for j in range(n_cols):
                    value = df_sorted.iloc[i, j]
                    # Only show non-zero values
                    if value > 0:
                        text_color = 'white' if value > 50 else 'black'
                        ax.text(j, i - 0.25, f"{value:.2f}%", 
                            ha='center', va='top', 
                            color=text_color, fontweight='bold',
                            fontsize=10)
        
        # Create legend elements and x-axis labels
        legend_elements = []
        x_labels = []
        
        for j, col_name in enumerate(df_sorted.columns):
            # Parse column name to extract predictor values
            parts = col_name.split()
            
            # Create tuple notation
            values = []
            var_names = []
            for part in parts:
                if "=" in part:
                    name, val = part.split("=")
                    values.append(val)
                    var_names.append(name)
            
            values_str = f"({', '.join(values)})"
            var_names_str = f"({', '.join(var_names)})"
            
            # Store both formats
            legend_elements.append(f"#{j+1}: {var_names_str} = {values_str}")
            x_labels.append(f"{values_str}")
        
        # Set x-axis labels based on legend style
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Set y-axis labels
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(df_sorted.index)
        
        # Add dots for predicted categories if predictions are available
        if has_predictions:
            for j, col_name in enumerate(df_sorted.columns):
                if col_name in df.predictions.columns:
                    pred_cat = df.predictions.loc["Predicted", col_name]
                    
                    # Find the row index for this category in the sorted dataframe
                    for i, idx in enumerate(df_sorted.index):
                        if idx.endswith(f"={pred_cat}"):
                            ax.plot(j, i, 'o', color='white', markersize=8, markerfacecolor='white')
                            break
        
        # Add title and labels
        pred_names = ", ".join(predictors_names)
        title = f"Bootstrap Prediction Percentages\n{response_name} Categories Given {pred_names}"
        ax.set_title(title)
        
        ax.set_xlabel(f"Category Combinations of {var_names_str}")
        ax.set_ylabel(f"{response_name} Categories")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Prediction Percentage (%)")
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
    
    # Attach the plotting method to the DataFrame
    summary_df.plot_prediction_heatmap = plot_prediction_heatmap
    
    return summary_df

@dataclass 
class CustomPermutationResult:
    """Container for permutation test results with visualization capabilities.
    
    Parameters
    ----------
    - metric_name : `str`
        Name of the metric being tested
    - observed_value : `float`
        Original observed value
    - p_value : `float`
        Permutation test p-value
    - null_distribution : `np.ndarray`
        Array of values under null hypothesis
    - permutation_tables : `np.ndarray`, optional
        Array of permuted contingency tables
    - histogram_fig : `plt.Figure`, optional
        Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    p_value: float
    null_distribution: np.ndarray
    permutation_tables: np.ndarray = None
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
                          random_state = None,
                          store_tables: bool = False) -> CustomPermutationResult:
    """Perform permutation test for (S)CCRAM measure.
    
    Parameters
    ----------
    - contingency_table : `np.ndarray`
        Input contingency table
    - predictors : `Union[List[int], int]`
        List of 1-indexed predictors axes for category prediction
    - response : `int`
        1-indexed target response axis for category prediction
    - scaled : `bool`, default=False
        Whether to use scaled CCRAM (SCCRAM)
    - alternative : `str`, default='greater'
        Alternative hypothesis ('greater', 'less', 'two-sided')
    - n_resamples : `int`, default=9999
        Number of permutations
    - random_state : `int`, optional
        Random state for reproducibility
    - store_tables : `bool`, default=False
        Whether to store the permuted contingency tables
        
    Returns
    -------
    `CustomPermutationResult`
        Test results including p-value, null distribution and tables
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
    
    # Store permutation tables if requested
    permutation_tables = None
    if store_tables:
        permutation_tables = np.zeros((n_resamples,) + contingency_table.shape)
    
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
            for i, batch_cases in enumerate(cases):
                table = gen_case_form_to_contingency(
                    batch_cases, 
                    shape=contingency_table.shape,
                    axis_order=all_axes
                )
                
                # Store table if requested and not the observed table (first one)
                if store_tables and permutation_tables is not None and i > 0 and i <= n_resamples:
                    permutation_tables[i-1] = table
                
                ccrvam = GenericCCRVAM.from_contingency_table(table)
                value = ccrvam.calculate_CCRAM(predictors, response, scaled)
                results.append(value)
            return np.array(results)
        else:
            table = gen_case_form_to_contingency(
                cases,
                shape=contingency_table.shape,
                axis_order=all_axes
            )
            ccrvam = GenericCCRVAM.from_contingency_table(table)
            return ccrvam.calculate_CCRAM(predictors, response, scaled)

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
        null_distribution=perm.null_distribution,
        permutation_tables=permutation_tables
    )
    
    result.plot_distribution(f'Null Distribution: {metric_name}')
    return result
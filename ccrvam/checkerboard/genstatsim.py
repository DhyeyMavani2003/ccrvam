import numpy as np
from scipy.stats import bootstrap, permutation_test
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from .genccrvam import GenericCCRVAM
from .utils import gen_contingency_to_case_form, gen_case_form_to_contingency

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class CustomBootstrapResult:
    """
    Container for bootstrap simulation (including confidence intervals) results
    with statistical visualization capabilities.
    
    Input Arguments
    --------------
    - `metric_name` : Name of the metric being bootstrapped
    - `observed_value` : Original observed value of the metric
    - `confidence_interval` : Lower and upper bounds for the bootstrap confidence interval
    - `bootstrap_distribution` : Array of bootstrapped values of the metric
    - `standard_error` : Standard error of the bootstrap distribution
    - `bootstrap_tables` : Array of bootstrapped contingency tables (optional)
    - `histogram_fig` : Matplotlib figure of distribution plot (optional)
    """
    
    metric_name: str 
    observed_value: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    standard_error: float
    bootstrap_tables: Optional[np.ndarray] = None
    histogram_fig: Optional[plt.Figure] = None

    def plot_distribution(
        self, 
        title: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot bootstrap distribution with observed value.

        Input Arguments
        --------------
        - `title` : Title of the plot (optional)

        Outputs
        -------
        Matplotlib figure of distribution plot
        
        Warnings/Errors
        --------------
        - `Exception` : If the plot cannot be created
        """
        # Check if bootstrap distribution data is available
        if self.bootstrap_distribution is None:
            print(f"Warning: Cannot plot distribution for {self.metric_name} as bootstrap_distribution data is missing.")
            self.histogram_fig = None # Ensure fig attribute is None
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_range = np.ptp(self.bootstrap_distribution)
            
            # Handle both exact zeros and very small ranges due to floating-point precision
            if data_range < 1e-10:
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

def bootstrap_ccram(
    contingency_table: np.ndarray,
    predictors: Union[List[int], int],
    response: int, 
    scaled: bool = False,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state: Optional[int] = None,
    store_tables: bool = False
) -> CustomBootstrapResult:
    """
    Perform bootstrap simulation and confidence intervals for (S)CCRAM.
    
    Input Arguments
    --------------
    - `contingency_table` : Input contingency table of frequency counts
    - `predictors` : List of 1-indexed predictor axes for (S)CCRAM calculation
    - `response` : 1-indexed target response variable axis for (S)CCRAM calculation
    - `scaled` : Whether to use scaled (S)CCRAM (default=False)
    - `n_resamples` : Number of bootstrap resamples (default=9999)
    - `confidence_level` : Confidence level for bootstrap confidence intervals (default=0.95)
    - `method` : Bootstrap CI method ('percentile', 'basic', 'BCa'); (default='percentile')
    - `random_state` : Random state for reproducibility (optional)
    - `store_tables` : Whether to store the bootstrapped contingency tables (default=False)
        
    Outputs
    -------
    Bootstrap result class containing bootstrap confidence interval, bootstrap estimates for the (S)CCRAM and bootstrap tables
        
    Warnings/Errors
    --------------
    - `ValueError` : If predictor or response axis is out of bounds
    """
    if not isinstance(predictors, list):
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
    
    # Check if bootstrap_distribution attribute exists (for compatibility with older scipy versions)
    bootstrap_distribution_values = getattr(res, 'bootstrap_distribution', None)
    if bootstrap_distribution_values is None:
        print(f"Warning: Bootstrap distribution data not available from scipy.stats.bootstrap result for {metric_name}. Plotting will be disabled.")

    result = CustomBootstrapResult(
        metric_name=metric_name,
        observed_value=observed_ccram,
        confidence_interval=res.confidence_interval,
        bootstrap_distribution=bootstrap_distribution_values,
        standard_error=res.standard_error,
        bootstrap_tables=bootstrap_tables
    )
    
    result.plot_distribution(f'Bootstrap Distribution: {metric_name}')
    return result

def _process_prediction_batch(args):
    """Helper function for parallel prediction processing."""
    batch_indices, cases, dims, all_axes, parsed_predictors, response, pred_combinations = args
    
    # batch_size = len(batch_indices)
    result = np.zeros((dims[response], len(pred_combinations)))
    
    # Process each bootstrap sample in the batch
    for indices in batch_indices:
        bootstrap_cases = cases[indices]
        bootstrap_table = gen_case_form_to_contingency(bootstrap_cases, shape=dims, axis_order=all_axes)
        ccrvam = GenericCCRVAM.from_contingency_table(bootstrap_table)
        
        # For each predictor combination, get the predicted category
        for i, combo in enumerate(pred_combinations):
            source_cats = [c-1 for c in combo]
            try:
                predicted = ccrvam._predict_category(
                    source_category=source_cats,
                    predictors=parsed_predictors,
                    response=response
                )
                result[predicted, i] += 1
            except Exception:
                continue
    
    return result

def bootstrap_predict_ccr_summary(
    table: np.ndarray,
    predictors: Union[List[int], int],
    predictors_names: Optional[List[str]] = None,
    response: Optional[int] = None,
    response_name: Optional[str] = None,
    n_resamples: int = 9999,
    random_state: Optional[int] = None,
    parallel: bool = True
) -> pd.DataFrame:
    """
    Compute bootstrap prediction matrix showing percentage predictions for each combination of predictor values in CCR analysis.
    
    Input Arguments
    --------------
    - `table` : Contingency table of frequency counts
    - `predictors` : List of predictor dimensions (1-indexed)
    - `predictors_names` : Names of predictor variables (optional)
    - `response` : Response variable dimension (1-indexed). If None, the last dimension is used.
    - `response_name` : Name of response variable (optional)
    - `n_resamples` : Number of bootstrap resamples (default=9999)
    - `random_state` : Random state for reproducibility (optional)
    - `parallel` : Whether to use parallel processing (default=True)
        
    Outputs
    -------
    CCR Prediction matrix post-bootstrap showing the percentage of the predicted category of the response variable for each combination of categories of the predictors.
    
    Warnings/Errors
    --------------
    - `ValueError` : If predictor or response axis is out of bounds
    
    Notes
    -----
    - The output is a pandas DataFrame with the percentage of the predicted category of the response variable for each combination of categories of the predictors.
    - The output also includes a method `plot_prediction_heatmap` to plot the prediction matrix as a heatmap.
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
        
    if not isinstance(predictors, list):
        predictors = [predictors]
    
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
    
    # Get all required axes in sorted order
    all_axes = sorted(parsed_predictors + [response])
    
    # Convert table to case form for resampling
    cases = gen_contingency_to_case_form(table)
    
    # Create all possible combinations of predictor values (1-indexed for output)
    pred_combinations = list(itertools.product(*[range(1, dim+1) for dim in pred_dims]))
    
    # Create column headers
    columns = [" ".join([f"{name}={val}" for name, val in zip(predictors_names, combo)]) 
              for combo in pred_combinations]
    
    # Create row labels (1-indexed for output)
    rows = [f"{response_name}={i+1}" for i in range(response_dim)]
    
    # Initialize result matrix with zeros
    result = np.zeros((response_dim, len(pred_combinations)))
    
    # Generate all bootstrap samples at once
    rng = np.random.RandomState(random_state)
    all_bootstrap_indices = [
        rng.choice(len(cases), size=len(cases), replace=True)
        for _ in range(n_resamples)
    ]
    
    if parallel:
        # Determine number of cores to use
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {n_jobs} cores for parallel processing")
        batch_size = max(1, n_resamples // n_jobs)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Prepare batches for parallel processing
            batches = []
            for i in range(0, n_resamples, batch_size):
                end_idx = min(i + batch_size, n_resamples)
                batch_indices = all_bootstrap_indices[i:end_idx]
                batches.append((batch_indices, cases, dims, all_axes, parsed_predictors, response, pred_combinations))
            
            # Process batches in parallel
            futures = [executor.submit(_process_prediction_batch, batch) for batch in batches]
            for future in as_completed(futures):
                result += future.result()
    else:
        # Process all samples sequentially
        batch = (all_bootstrap_indices, cases, dims, all_axes, parsed_predictors, response, pred_combinations)
        result = _process_prediction_batch(batch)
    
    # Convert counts to percentages using vectorized operations
    col_sums = result.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentages = np.where(col_sums > 0, 
                             (result / col_sums[:, np.newaxis].T) * 100, 
                             0)
    
    # Create DataFrame
    summary_df = pd.DataFrame(percentages, index=rows, columns=columns)
    
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
    
    def plot_predictions_summary(
        prediction_matrix: pd.DataFrame = summary_df, 
        figsize: Optional[Tuple[int, int]] = None, 
        show_values: bool = True,
        show_indep_line: bool = True,
        cmap: str = 'Blues', 
        save_path: Optional[str] = None, 
        dpi: Optional[int] = 300,
        plot_type: str = 'heatmap'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot prediction percentages as either a heatmap or bubble plot visualization.
        
        Input Arguments
        --------------
        - `prediction_matrix` : DataFrame of prediction percentages (default=summary_df)
        - `figsize` : Tuple of figure size (width, height) (optional)
        - `show_values` : Whether to show percentage values (default=True)
        - `show_indep_line` : Whether to show dotted line for predictions under joint independence (default=True)
        - `cmap` : Colormap for visualization (default='Blues')
        - `save_path` : Path to save the plot (optional)
        - `dpi` : Resolution for saved image (default=300) (optional)
        - `plot_type` : Type of plot to generate ('heatmap' or 'bubble') (default='heatmap')
            
        Outputs
        -------
        Tuple of Matplotlib figure and axes objects for the plot
        """
        # Get data dimensions
        n_rows, n_cols = prediction_matrix.shape
        
        # Set figure size based on data dimensions
        if figsize is None:
            figsize = (max(8, n_cols * 1.2), 
                    max(6, n_rows * 1.2))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort the DataFrame by index in descending order to get categories from highest to lowest
        prediction_matrix_sorted = prediction_matrix.sort_index(ascending=False)
        
        # Check if we have predictions attribute
        has_predictions = hasattr(prediction_matrix, 'predictions')
        
        # Create legend elements and x-axis labels
        legend_elements = []
        x_labels = []
        
        for j, col_name in enumerate(prediction_matrix_sorted.columns):
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
        
        if plot_type == 'heatmap':
            # Create heatmap with sorted data
            im = ax.imshow(prediction_matrix_sorted.values, cmap=cmap, aspect='auto')
            
            # Add text values if requested
            if show_values:
                for i in range(n_rows):
                    for j in range(n_cols):
                        value = prediction_matrix_sorted.iloc[i, j]
                        # Only show non-zero values
                        if value > 0:
                            text_color = 'white' if value > 50 else 'black'
                            ax.text(j, i - 0.25, f"{value:.2f}%", 
                                ha='center', va='top', 
                                color=text_color, fontweight='bold',
                                fontsize=10)
            
            # Set x-axis labels based on legend style
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Set y-axis labels
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(prediction_matrix_sorted.index)
            
            # Add dots for predicted categories if predictions are available
            if has_predictions:
                for j, col_name in enumerate(prediction_matrix_sorted.columns):
                    if col_name in prediction_matrix.predictions.columns:
                        pred_cat = prediction_matrix.predictions.loc["Predicted", col_name]
                        
                        # Find the row index for this category in the sorted dataframe
                        for i, idx in enumerate(prediction_matrix_sorted.index):
                            if idx.endswith(f"={pred_cat}"):
                                ax.plot(j, i, 'o', color='white', markersize=8, markerfacecolor='white')
                                break
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Prediction Percentage (%)")
            
        elif plot_type == 'bubble':
            # Create bubble plot
            x = np.arange(n_cols)
            y = np.arange(n_rows)
            X, Y = np.meshgrid(x, y)
            
            # Get values and normalize for bubble sizes
            values = prediction_matrix_sorted.values
            sizes = values.flatten() * 100  # Scale up for better visibility
            colors = values.flatten()
            
            # Create scatter plot with varying size and color
            scatter = ax.scatter(X.flatten(), Y.flatten(), 
                               s=sizes, c=colors, cmap=cmap,
                               alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Add text values if requested
            if show_values:
                for i in range(n_rows):
                    for j in range(n_cols):
                        value = prediction_matrix_sorted.iloc[i, j]
                        # Only show non-zero values
                        if value > 0:
                            text_color = 'white' if value > 50 else 'black'
                            ax.text(j, i + 0.25, f"{value:.2f}%", 
                                ha='center', va='center', 
                                color=text_color, fontweight='bold',
                                fontsize=9)
            
            # Set x-axis labels
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Set y-axis labels
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(prediction_matrix_sorted.index)
            
            # Add dots for predicted categories if predictions are available
            if has_predictions:
                for j, col_name in enumerate(prediction_matrix_sorted.columns):
                    if col_name in prediction_matrix.predictions.columns:
                        pred_cat = prediction_matrix.predictions.loc["Predicted", col_name]
                        
                        # Find the row index for this category in the sorted dataframe
                        for i, idx in enumerate(prediction_matrix_sorted.index):
                            if idx.endswith(f"={pred_cat}"):
                                ax.plot(j, i, 'o', color='white', markersize=8, 
                                      markerfacecolor='white', markeredgecolor='black')
                                break
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Prediction Percentage (%)")
            
            # Set aspect ratio to 'equal' for better bubble visualization
            ax.set_aspect('equal')
            
            # Adjust y-axis limits to prevent bubble cutoff
            ax.set_ylim(-0.5, n_rows - 0.5)
            
        else:
            raise ValueError("plot_type must be either 'heatmap' or 'bubble'")
        
        # Add dotted line for independence predictions if requested
        if show_indep_line and has_predictions:
            ccrvam = GenericCCRVAM.from_contingency_table(table)
            # Convert to plot y-coordinate (top-down ordering)
            response_cats = ccrvam.P.shape[response]
            pred_cat_under_indep = ccrvam.get_prediction_under_indep(response+1)
            indep_y_pos = response_cats - pred_cat_under_indep
            ax.axhline(y=indep_y_pos, color='red', linestyle='--', linewidth=1.1, alpha=0.9)
        
        # Add title and labels
        pred_names = ", ".join(predictors_names)
        title_base = f"Bootstrap Prediction Percentages\n{response_name} Categories Given {pred_names}"
        
        # Add information about dotted line if it's shown
        if show_indep_line:
            title = f"{title_base}\nDotted line: predicted category under joint independence"
        else:
            title = title_base
            
        ax.set_title(title)
        
        ax.set_xlabel(f"Category Combinations of {var_names_str}")
        ax.set_ylabel(f"{response_name} Categories")
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
        return fig, ax
    
    # Transpose the DataFrame for better view on the user side
    summary_df = np.transpose(summary_df)
    
    # Add predictions DataFrame as an attribute to the summary DataFrame
    # This will allow us to access the predictions in the same format as summary_df
    summary_df.predictions = np.transpose(pred_df)
    
    # Attach the plotting and saving methods to the DataFrame
    summary_df.plot_predictions_summary = plot_predictions_summary
    
    return summary_df

def save_predictions(
    prediction_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
    format: str = 'csv',
    decimal_places: int = 2
) -> None:
    """
    Save prediction results generated by `bootstrap_predict_ccr_summary()` to a file.
    
    Input Arguments
    --------------
    - `prediction_matrix` : DataFrame containing prediction results generated by `bootstrap_predict_ccr_summary()`
    - `save_path` : Path to save the output file
    - `format` : Output format ('csv' or 'txt')
    - `decimal_places` : Number of decimal places for prediction percentages results from `bootstrap_predict_ccr_summary()`
    
    Outputs
    -------
    None (saves file to disk)
    
    Warnings/Errors
    --------------
    - `ValueError` : If save_path is not specified
    """
    if save_path is None:
        raise ValueError("save_path must be specified")
        
    # Create output directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Unpack the DataFrame by transposing it
    predictions = np.transpose(prediction_matrix.predictions)
    prediction_matrix = np.transpose(prediction_matrix)
    
    # Prepare data for output
    output_data = {}
    
    # Add predictor combinations and their data
    for col in prediction_matrix.columns:
        # Store the column name as is
        output_data[col] = {}

        output_data[col]['Predicted_Category'] = int(predictions.loc['Predicted', col])
        
        percentages = prediction_matrix[col].round(decimal_places)
        for idx, pct in percentages.items():
            # Use the index name directly
            output_data[col][idx] = pct
    
    # Save based on format
    if format.lower() == 'csv':
        pd.DataFrame(output_data).T.to_csv(save_path)
    elif format.lower() == 'txt':
        with open(save_path, 'w') as f:
            for combo, data in output_data.items():
                f.write(f"Predictor Combination: {combo}\n")
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'txt'")

@dataclass 
class CustomPermutationResult:
    """
    Container for permutation simulation (including test) results
    with statistical visualization capabilities.
    
    Input Arguments
    --------------
    - `metric_name` : Name of the (S)CCRAM being tested   
    - `observed_value` : Original observed value of the (S)CCRAM
    - `p_value` : Permutation test p-value
    - `null_distribution` : Array of the values of the (S)CCRAM computed for the permuted contingency tables
    - `permutation_tables` : (Optional) Array of permuted contingency tables generated under the null hypothesis (no regression association)
    - `histogram_fig` : (Optional) Matplotlib figure of distribution plot
    """
    metric_name: str
    observed_value: float
    p_value: float
    null_distribution: np.ndarray
    permutation_tables: np.ndarray = None
    histogram_fig: plt.Figure = None

    def plot_distribution(
        self,
        title: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot null distribution with observed value.
        
        Input Arguments
        --------------
        - `title` : Title of the plot (optional)

        Outputs
        -------
        Matplotlib figure of distribution plot
        """
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

def permutation_test_ccram(
    contingency_table: np.ndarray,
    predictors: Union[List[int], int],
    response: int,
    scaled: bool = False,
    alternative: str = 'greater',
    n_resamples: int = 9999,
    random_state: Optional[int] = None,
    store_tables: bool = False
) -> CustomPermutationResult:
    """
    Perform permutation simulation and test for (S)CCRAM.
    
    Input Arguments
    --------------
    - `contingency_table` : Input contingency table of frequency counts
    - `predictors` : List of 1-indexed predictors axes for (S)CCRAM calculation
    - `response` : 1-indexed target response axis for (S)CCRAM calculation
    - `scaled` : Whether to use scaled (S)CCRAM (default=False)
    - `alternative` : Alternative hypothesis ('greater', 'less', 'two-sided') (default='greater')
    - `n_resamples` : Number of permutations (default=9999)
    - `random_state` : Random state for reproducibility (optional)
    - `store_tables` : Whether to store the permuted contingency tables (default=False)
        
    Outputs
    -------
    Test results including Monte Carlo permutation p-value, (S)CCRAM values computed for the permuted contingency tables, 
    and (optionally) permuted contingency tables generated under the null hypothesis (no regression association)
        
    Warnings/Errors
    --------------
    - `ValueError` : If predictor or response variable axis is out of bounds
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
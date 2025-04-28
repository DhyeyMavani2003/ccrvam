import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import os
from .utils import gen_case_form_to_contingency

class GenericCCRVAM:
    """Central Generic Checkerboard Copula Regression, Visualization and Association Measure (CCRVAM) class object."""
    @classmethod
    def from_contingency_table(
        cls: 'GenericCCRVAM',
        contingency_table: np.ndarray
    ) -> 'GenericCCRVAM':
        """
        Create a CCRVAM object instance from a multi-dimensional contingency table.

        Input Arguments
        --------------
        - `contingency_table` : A contingency table of frequency counts (multi-dimensional numpy array)

        Outputs
        -------
        A new GenericCCRVAM object instance initialized with the probability matrix, which will allow for further statistical analysis of the data.

        Warnings/Errors
        --------------
        - `ValueError` : If the input table contains negative values or all zeros
        """ 
        if np.any(contingency_table < 0):
            raise ValueError("Contingency table cannot contain negative values")
            
        total_count = contingency_table.sum()
        if total_count == 0:
            raise ValueError("Contingency table cannot be all zeros")
            
        P = contingency_table / total_count
        return cls(P)
    
    @classmethod
    def from_cases(
        cls: 'GenericCCRVAM',
        cases: np.ndarray,
        dimension: tuple
    ) -> 'GenericCCRVAM':
        """
        Create a CCRVAM object instance from the case form data.

        Input Arguments
        --------------
        - `cases` : A 2D array where each row represents a case array of observed values for each categorical variable.
                   Each column corresponds to a different variable, and the values in each column represent the 
                   category indices for that variable (1-indexed).
        - `dimension` : A tuple specifying the number of categories for each variable in the same order as the columns
                       in `cases`. For example, if `cases` has 3 columns representing variables A, B, and C with 
                       2, 3, and 4 categories respectively, then `dimension` should be (2,3,4).

        Outputs
        -------
        A new GenericCCRVAM object instance initialized with the probability matrix, which will allow for further 
        statistical analysis of the data.

        Warnings/Errors
        --------------
        - `ValueError` : If the input cases are not 2-dimensional or if the dimension tuple does not match the 
                        number of variables and their categories in the cases data.

        Example
        -------
        >>> cases = np.array([[1,1,1], [1,2,1], [2,1,2]])  # 3 cases with 3 variables
        >>> dimension = (2,2,2)  # Each variable has 2 categories
        >>> ccrvam = GenericCCRVAM.from_cases(cases, dimension)
        """
        if cases.ndim != 2:
            raise ValueError("Cases must be a 2D array")
            
        if cases.shape[1] != len(dimension):
            raise ValueError("Dimension tuple must match number of variables")
            
        # Convert from 1-indexed input to 0-indexed categorical cases
        cases -= 1
        
        contingency_table = gen_case_form_to_contingency(cases, dimension)
        return cls.from_contingency_table(contingency_table)
    
    def __init__(
        self,
        P: np.ndarray
    ):
        """Initialization with joint probability matrix P for statistical analysis."""
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("P must contain values in [0,1]")
            
        if not np.allclose(P.sum(), 1.0):
            raise ValueError("P must sum to 1")
            
        self.P = P
        self.ndim = P.ndim
        self.dimension = P.shape
        
        # Calculate and store marginals for each axis
        self.marginal_pmfs = {}
        self.marginal_cdfs = {}
        self.scores = {}
        
        for axis in range(self.ndim):
            # Calculate marginal PMF
            pmf = P.sum(axis=tuple(i for i in range(self.ndim) if i != axis))
            self.marginal_pmfs[axis] = pmf
            
            # Calculate marginal CDF
            cdf = np.insert(np.cumsum(pmf), 0, 0)
            self.marginal_cdfs[axis] = cdf
            
            # Calculate scores
            self.scores[axis] = self._calculate_scores(cdf)
            
        # Store conditional PMFs
        self.conditional_pmfs = {}
        
    def calculate_CCRAM(
        self,
        predictors: Union[int, list],
        response: int,
        scaled: bool = False
    ) -> float:
        """
        Calculate CCRAM with multiple conditioning axes.
        
        Input Arguments
        --------------
        - `predictors` : List of 1-indexed predictors axes for regression association
        - `response` : 1-indexed target response variable axis for regression association
        - `scaled` : Whether to return scaled or normalized CCRAM statistical measure (default: False)
            
        Outputs
        -------
        (Scaled) CCRAM value for the given predictors and response variable
        
        Warnings/Errors
        --------------
        - `ValueError` : If the response variable axis is out of bounds for the array dimension
        - `ValueError` : If the predictors contain an axis that is out of bounds for the array dimension
        """
        if not isinstance(predictors, list):
            predictors = [predictors]
            
        # Input validation
        parsed_predictors = [pred_axis - 1 for pred_axis in predictors]
        parsed_response = response - 1
        
        if parsed_response >= self.ndim:
            raise ValueError(f"parsed response {parsed_response} is out of bounds for array of dimension {self.ndim}")
        
        for axis in parsed_predictors:
            if axis >= self.ndim:
                raise ValueError(f"parsed predictors contains {axis} which is out of bounds for array of dimension {self.ndim}")
        
        # Calculate marginal pmf of predictors
        sum_axes = tuple(set(range(self.ndim)) - set(parsed_predictors))
        preds_pmf_prob = self.P.sum(axis=sum_axes)
        
        # Calculate regression values for each combination
        weighted_expectation = 0.0
        
        for idx in np.ndindex(preds_pmf_prob.shape):
            u_values = [self.marginal_cdfs[axis][idx[parsed_predictors.index(axis)] + 1] 
                        for axis in parsed_predictors]
            
            regression_value = self._calculate_regression_batched(
                target_axis=parsed_response,
                given_axes=parsed_predictors,
                given_values=u_values
            )[0]
            
            weighted_expectation += preds_pmf_prob[idx] * (regression_value - 0.5) ** 2
        
        ccram = 12 * weighted_expectation
        
        if not scaled:
            return ccram
            
        sigma_sq_S = self._calculate_sigma_sq_S(parsed_response)
        if sigma_sq_S < 1e-10:
            return 1.0 if ccram >= 1e-10 else 0.0
        return ccram / (12 * sigma_sq_S)

    def get_predictions_ccr(
        self,
        predictors: list,
        response: int,
        variable_names: Union[dict, None] = None
    ) -> pd.DataFrame:
        """
        Get predictions of response variable categories conditioned on multiple predictor variables.
        
        Input Arguments
        --------------
        - `predictors` : List of 1-indexed predictors axes for category prediction
        - `response` : 1-indexed target response variable axis for category prediction
        - `variable_names` : Dictionary mapping 1-indexed variable indices to names (default: None)
            
        Outputs
        -------
        DataFrame containing the predicted category of the response variable for each combination of categories of the predictors
        
        Notes
        -----
        The DataFrame contains columns for each combination of categories of the predictors and the corresponding predicted category of the response variable.
        The categories are 1-indexed.
        
        Warnings/Errors
        --------------
        - `ValueError` : If the response variable axis is out of bounds for the array dimension
        - `ValueError` : If the predictors contain an axis that is out of bounds for the array dimension
        """
        # Flag to hide response default name if variable_names is not provided
        hide_response_name_flag = False
        if variable_names is None:
            hide_response_name_flag = True
            variable_names = {i+1: f"X{i+1}" for i in range(self.ndim)}
        
        # Input validation
        parsed_predictors = []
        for pred_axis in predictors:
            if pred_axis < 1 or pred_axis > self.ndim:
                raise ValueError(f"Predictor axis {pred_axis} is out of bounds")
            parsed_predictors.append(pred_axis - 1)
        parsed_response = response - 1
        
        # Create meshgrid of source categories
        source_dims = [self.P.shape[axis] for axis in parsed_predictors]
        source_categories = [np.arange(dim) for dim in source_dims]
        mesh = np.meshgrid(*source_categories, indexing='ij')
        
        # Flatten for prediction
        flat_categories = [m.flatten() for m in mesh]
        
        # Get predictions
        predictions = self._predict_category_batched_multi(
            source_categories=flat_categories,
            predictors=parsed_predictors,
            response=parsed_response
        )
        
        # Create DataFrame
        result = pd.DataFrame()
        for axis, cats in zip(parsed_predictors, flat_categories):
            result[f'{variable_names[axis+1]} Category'] = cats + 1
            
        response_name = variable_names[response] if not hide_response_name_flag else "Response"
        result[f'Predicted {response_name} Category'] = predictions + 1
        
        return result
    
    def get_prediction_under_indep(
        self,
        response: int
    ) -> int:
        """
        Calculate the predicted category under joint independence between the response variables and predictors.
        
        The CCR value equals 0.5 under the assumption of joint independence between the 
        response variable and all predictor variables.
        
        Input Arguments
        --------------
        - `response` : 1-indexed target response variable axis
            
        Outputs
        -------
        The predicted category (1-indexed) for the response variable under joint independence
        
        Notes
        -----
        This prediction serves as an important reference point when interpreting
        CCR prediction results, as it represents what would be predicted if there
        were no association between the predictors and the response variable.
        
        Warnings/Errors
        --------------
        - `ValueError` : If the response variable axis is out of bounds for the array dimension
        """
        parsed_response = response - 1
        
        if parsed_response < 0 or parsed_response >= self.ndim:
            raise ValueError(f"Response variable axis {response} is out of bounds")
        
        # Under independence, the regression value is 0.5 according to Proposition 2.1(c)
        independence_regression_value = 0.5
        
        # Get the predicted category (0-indexed)
        predicted_cat = self._get_predicted_category(
            independence_regression_value, 
            self.marginal_cdfs[parsed_response]
        )
        
        # Return 1-indexed category
        return predicted_cat + 1
    
    def calculate_ccs(
        self,
        var_index: int
    ) -> np.ndarray:
        """
        Calculate checkerboard scores for the specified variable index.
        
        Input Arguments
        --------------
        - `var_index` : 1-Indexed axis of the variable for which to calculate scores
            
        Outputs
        -------
        Array containing checkerboard scores for the given variable index
        
        Warnings/Errors
        --------------
        - `ValueError` : If the axis is out of bounds for the array dimension
        """
        parsed_axis = var_index - 1
        return self.scores[parsed_axis]
    
    def calculate_variance_ccs(
        self,
        var_index: int
    ) -> float:
        """
        Calculate the variance of the checkerboard score for the specified variable index.

        Input Arguments
        --------------
        - `var_index` : 1-Indexed axis of the variable for which to calculate variance of the checkerboard score
            
        Outputs
        -------
        - `float` : Variance of the checkerboard score for the given variable index
        
        Warnings/Errors
        --------------
        - `ValueError` : If the variable index is out of bounds for the array dimension
        """
        parsed_axis = var_index - 1
        return self._calculate_sigma_sq_S_vectorized(parsed_axis)
    
    def plot_ccr_predictions(
        self,
        predictors: list,
        response: int,
        variable_names: Union[dict, None] = None,
        legend_style: str = 'side',
        show_indep_line: bool = True,
        figsize: Union[tuple, None] = None,
        save_path: Union[str, None] = None,
        dpi: int = 300
    ) -> None:
        """
        Plot CCR predictions as a visualization.
        
        Input Arguments
        --------------
        - `predictors` : List of 1-indexed predictor axes
        - `response` : 1-indexed response variable axis
        - `variable_names` : Dictionary mapping indices to variable names (default: None)
        - `legend_style` : How to display combinations of categories of predictors: 'side' (default) or 'xaxis'
        - `show_indep_line` : Whether to show the prediction under joint independence between the response variable and all the predictors (default: True)
        - `figsize` : Figure size (width, height)
        - `save_path` : Path to save the plot (e.g. 'plots/ccr_pred.pdf')
        - `dpi` : Resolution for saving raster images (png, jpg)
        
        Outputs
        -------
        None (Plot is displayed or saved to file as per user preferences and settings)
        
        Warnings/Errors
        --------------
        - `ValueError` : If the response variable axis is out of bounds for the array dimension
        - `ValueError` : If the predictors contain an axis that is out of bounds for the array dimension
        """
        
        # Flag to hide response default name if variable_names is not provided
        hide_response_name_flag = False
        if variable_names is None:
            hide_response_name_flag = True
            variable_names = {i+1: f"X{i+1}" for i in range(self.ndim)}
            
        # Get predictions DataFrame
        predictions_df = self.get_predictions_ccr(predictors, response, variable_names)
        
        # Get the number of categories for the response variable and reverse order
        response_cats = self.P.shape[response-1]
        
        # Get all possible combinations of predictor categories
        pred_cat_columns = [col for col in predictions_df.columns if "Category" in col and "Predicted" not in col]
        
        # Create a matrix for the heatmap (rows=response categories, columns=predictor combinations)
        heatmap_data = np.zeros((response_cats, len(predictions_df)))
        
        # Fill in the predicted categories
        for i, pred_cat in enumerate(predictions_df.iloc[:, -1]):
            heatmap_data[int(pred_cat)-1, i] = 1  # Mark the predicted category with 1
        
        # Flip matrix vertically
        heatmap_data = np.flip(heatmap_data, axis=0)
        
        # Determine a good figure size based on the number of combinations
        if figsize is None:
            n_combos = len(predictions_df)
            width = max(8, min(n_combos * 0.3, 14))  # Limit maximum width
            height = max(6, response_cats * 0.7)
            figsize = (width, height)
                
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot white background without grid
        ax.imshow(heatmap_data, aspect='auto', cmap='binary', 
                interpolation='nearest', alpha=0.0)  # Completely transparent background
        
        # Set y-axis labels (response categories in descending order)
        ax.set_yticks(range(response_cats))
        ax.set_yticklabels([f"{response_cats-i}" for i in range(response_cats)])
        
        # Create x-axis labels and legend elements
        legend_elements = []
        x_labels = []
        for i, row in predictions_df[pred_cat_columns].iterrows():
            # Extract values and variable names
            values = [str(int(row[col])) for col in pred_cat_columns]
            var_names = [col.rsplit(' Category', 1)[0] for col in pred_cat_columns]
            
            # Create tuple notation
            values_str = f"({', '.join(values)})"
            var_names_str = f"({', '.join(var_names)})"
            
            # Store both formats
            legend_elements.append(f"#{i+1}: {var_names_str} = {values_str}")
            x_labels.append(f"{values_str}" if legend_style == 'xaxis' else f"{i+1}")
        
        # Set x-axis labels
        ax.set_xticks(range(len(predictions_df)))
        ax.set_xticklabels(x_labels)
        if legend_style == 'xaxis':
            plt.xticks(rotation=45, ha='right')
        
        # Add circles to show predicted categories
        for col in range(len(predictions_df)):
            pred_cat = int(predictions_df.iloc[col, -1])
            y_pos = response_cats - pred_cat
            ax.plot(col, y_pos, 'o', color='black', 
                    markersize=8, markerfacecolor='black')
        
        # Set titles and labels
        ax.set_xlabel("Predictor Combination Index" if legend_style == 'side' else f"Category Combinations of {var_names_str}")
        response_name = variable_names[response] if not hide_response_name_flag else "Response"
        ax.set_ylabel(f"Predicted {response_name} Category")
        
        pred_names = [variable_names[p] for p in predictors]
        pred_names_str = ", ".join(pred_names)
        ax.set_title(f"Predicted {response_name} Categories\nBased on {pred_names_str}")
        
        # Add horizontal line showing prediction under joint independence
        if show_indep_line:
            # Compute predicted category under joint independence (u=0.5)
            pred_cat_under_indep = self.get_prediction_under_indep(response)
            
            # Convert to plot y-coordinate (top-down ordering)
            indep_y_pos = response_cats - pred_cat_under_indep
            
            # Draw horizontal line across plot and add annotation
            ax.axhline(y=indep_y_pos, color='blue', linestyle='--', alpha=0.7, 
                    label=f"Prediction under joint independence: {pred_cat_under_indep}")
            
            # Add text label at right edge
            ax.text(len(predictions_df)-1, indep_y_pos + 0.3, 
                    f"Prediction under joint independence: {pred_cat_under_indep}", 
                    color='blue', ha='right', va='bottom', fontsize=9)
        
        # Create a legend with combination mappings
        legend_title = "Predictor Combinations:"
        
        # Add legend if using side style
        if legend_style == 'side':
            # Calculate figure size based on number of combinations
            if len(legend_elements) > 15:
                # Calculate the height needed for the legend
                legend_height = min(12, len(legend_elements) * 0.3 + 0.5)  # Increased height ratio
                legend_fig, legend_ax = plt.subplots(figsize=(6, legend_height))
                legend_ax.axis('off')
                
                # Create legend entries
                legend_text = [legend_title]
                legend_text.extend(legend_elements)
                
                # Display as text in the legend figure with smaller line spacing
                y_pos = 0.98  # Start slightly below top
                line_height = 0.95 / max(len(legend_text), 15)  # Adjusted line height
                
                legend_ax.text(0.05, y_pos, legend_title, fontweight='bold', 
                            va='top', transform=legend_ax.transAxes)
                y_pos -= line_height * 1.2  # Extra space after title
                
                # Add all combinations to legend
                for entry in legend_elements:
                    legend_ax.text(0.05, y_pos, entry, va='top', fontsize=9,
                                transform=legend_ax.transAxes)
                    y_pos -= line_height
                    
                legend_fig.tight_layout()
            else:
                # For fewer combinations, use a standard legend on the main plot
                handles = [plt.Line2D([], [], marker='none', color='none')] * len(legend_elements)
                ax.legend(handles, legend_elements, title=legend_title,
                        loc='center left', bbox_to_anchor=(1.05, 0.5), 
                        fontsize='small', frameon=False)
        
        # Adjust layout and save plot
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Save with appropriate format
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            # Save legend to separate file if it exists
            if len(legend_elements) > 15:
                legend_path = save_path.rsplit('.', 1)
                legend_path = f"{legend_path[0]}_legend.{legend_path[1]}"
                legend_fig.savefig(legend_path, dpi=dpi, bbox_inches='tight')
    
    def _calculate_conditional_pmf(self, target_axis, given_axes):
        """Internal helper function: Calculate conditional probability mass function of the response variable given the predictors."""
        if not isinstance(given_axes, (list, tuple)):
            given_axes = [given_axes]
                
        # Key for storing in conditional_pmfs dict
        key = (target_axis, tuple(sorted(given_axes)))
        
        # Return cached result if available
        if key in self.conditional_pmfs:
            return self.conditional_pmfs[key]
        
        # Calculate axes to sum over (marginalize)
        all_axes = set(range(self.ndim))
        keep_axes = set([target_axis] + list(given_axes))
        sum_axes = tuple(sorted(all_axes - keep_axes))
        
        # Create mapping of old axes to new positions
        old_to_new = {}
        new_pos = 0
        for axis in sorted(keep_axes):
            old_to_new[axis] = new_pos
            new_pos += 1
        
        # Calculate joint probability P(target,given)
        if sum_axes:
            joint_prob = self.P.sum(axis=sum_axes)
        else:
            joint_prob = self.P
        
        # Move target axis to first position
        target_new_pos = old_to_new[target_axis]
        joint_prob_reordered = np.moveaxis(joint_prob, target_new_pos, 0)
        
        # Calculate marginal probability P(given)
        marginal_prob = joint_prob_reordered.sum(axis=0, keepdims=True)
        
        # Calculate conditional probability P(target|given)
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_prob = np.divide(
                joint_prob_reordered, 
                marginal_prob,
                out=np.zeros_like(joint_prob_reordered),
                where=marginal_prob!=0
            )
        
        # Move axis back to original position
        conditional_prob = np.moveaxis(conditional_prob, 0, target_new_pos)
        
        # Store axis mapping with result
        self.conditional_pmfs[key] = (conditional_prob, old_to_new)
        return conditional_prob, old_to_new

    def _calculate_regression_batched(self, target_axis, given_axes, given_values):
        """Internal helper function: Vectorized regression calculation for multiple predictor variables."""
        if not isinstance(given_axes, (list, tuple)):
            given_axes = [given_axes]
            given_values = [given_values]
        
        # Convert scalar inputs to arrays
        given_values = [np.atleast_1d(values) for values in given_values]
        
        # Find intervals for all values in each axis
        intervals = []
        for axis, values in zip(given_axes, given_values):
            breakpoints = self.marginal_cdfs[axis][1:-1]
            intervals.append(np.searchsorted(breakpoints, values, side='left'))
        
        # Get conditional PMF and axis mapping
        conditional_pmf, axis_mapping = self._calculate_conditional_pmf(
            target_axis=target_axis,
            given_axes=given_axes
        )
        
        # Prepare output array
        n_points = len(given_values[0])
        results = np.zeros(n_points, dtype=float)
        
        # Calculate unique interval combinations
        unique_intervals = np.unique(np.column_stack(intervals), axis=0)
        
        # Calculate regression for each unique combination
        for interval_combo in unique_intervals:
            mask = np.all([intervals[i] == interval_combo[i] 
                        for i in range(len(intervals))], axis=0)
            
            # Select appropriate slice using mapped positions
            slicing = [slice(None)] * conditional_pmf.ndim
            for idx, axis in enumerate(given_axes):
                new_pos = axis_mapping[axis]
                slicing[new_pos] = interval_combo[idx]
                
            pmf_slice = conditional_pmf[tuple(slicing)]
            regression_value = np.sum(pmf_slice * self.scores[target_axis])
            results[mask] = regression_value
            
        return results
    
    def _calculate_scores(self, marginal_cdf):
        """Internal helper function: Calculate checkerboard scores from marginal CDF."""
        return [(marginal_cdf[j-1] + marginal_cdf[j])/2 
                for j in range(1, len(marginal_cdf))]
    
    def _lambda_function(self, u, ul, uj):
        """Internal helper function: Calculate lambda function for checkerboard copula construction through bilinear interpolation. (Wei and Kim, 2021)"""
        if u <= ul:
            return 0.0
        elif u >= uj:
            return 1.0
        else:
            return (u - ul) / (uj - ul)
        
    def _get_predicted_category(self, regression_value, marginal_cdf):
        """Internal helper function: Get predicted category based on the calculated regression value."""
        return np.searchsorted(marginal_cdf[1:-1], regression_value, side='left')

    def _get_predicted_category_batched(self, regression_values, marginal_cdf):
        """Internal helper function: Get predicted categories for multiple calculated regression values."""
        return np.searchsorted(marginal_cdf[1:-1], regression_values, side='left')
    
    def _calculate_sigma_sq_S(self, axis):
        """Internal helper function: Calculate variance of the checkerboard copula score for given axis."""
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Calculate each term in the sum
        terms = []
        for i in range(len(self.marginal_pmfs[axis])):
            if i < len(u_prev) and i < len(u_next):
                term = u_prev[i] * u_next[i] * self.marginal_pmfs[axis][i]
                terms.append(term)
        
        # Calculate sigma_sq_S
        sigma_sq_S = sum(terms) / 4.0
        return sigma_sq_S

    def _calculate_sigma_sq_S_vectorized(self, axis):
        """Internal helper function: Calculate variance of the checkerboard score using vectorized operations."""
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pmfs[axis]
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        return sigma_sq_S
    
    def _predict_category(self, source_category, predictors, response):
        """Internal helper function: Predict the category of the response variable given given combination of categories of predictors."""
        if not isinstance(source_category, (list, tuple)):
            source_category = [source_category]
        if not isinstance(predictors, (list, tuple)):
            predictors = [predictors]
            
        # Get corresponding u values for each axis
        u_values = [
            self.marginal_cdfs[axis][cat + 1]
            for axis, cat in zip(predictors, source_category)
        ]
        
        # Get regression value
        u_target = self._calculate_regression_batched(
            target_axis=response,
            given_axes=predictors,
            given_values=u_values
        )
        
        # Get predicted category
        predicted_category = self._get_predicted_category(u_target, self.marginal_cdfs[response])
        
        return predicted_category
    
    def _predict_category_batched_multi(
        self, 
        source_categories, 
        predictors, 
        response
    ):
        """Internal helper function: Vectorized prediction with multiple predictor variables."""
        if not isinstance(predictors, (list, tuple)):
            predictors = [predictors]

        # Get corresponding u values
        u_values = [
            self.marginal_cdfs[axis][cats + 1]
            for axis, cats in zip(predictors, source_categories)
        ]
        
        # Calculate regression values
        u_target_values = self._calculate_regression_batched(
            target_axis=response,
            given_axes=predictors,
            given_values=u_values
        )
        
        return self._get_predicted_category_batched(
            u_target_values,
            self.marginal_cdfs[response]
        )
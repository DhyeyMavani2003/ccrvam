import numpy as np
import pandas as pd
from .utils import gen_case_form_to_contingency

class GenericCCRVAM:
    @classmethod
    def from_contingency_table(cls, contingency_table):
        """
        Create a CCRVAM object instance from a contingency table.

        Parameters
        ----------
        contingency_table : numpy.ndarray
            A 2D contingency table of counts/frequencies.

        Returns
        -------
        CCRVAM
            A new instance initialized with the probability matrix.

        Raises
        ------
        ValueError
            If the input table contains negative values or all zeros.
            If the input table is not 2-dimensional.

        Examples
        --------
        >>> table = np.array([
            [0, 0, 20],
            [0, 10, 0],
            [20, 0, 0],
            [0, 10, 0],
            [0, 0, 20]
        ])
        >>> copula = GenericCCRVAM.from_contingency_table(table)
        """
        if not isinstance(contingency_table, np.ndarray):
            contingency_table = np.array(contingency_table)
            
        if np.any(contingency_table < 0):
            raise ValueError("Contingency table cannot contain negative values")
            
        total_count = contingency_table.sum()
        if total_count == 0:
            raise ValueError("Contingency table cannot be all zeros")
            
        P = contingency_table / total_count
        return cls(P)
    
    @classmethod
    def from_cases(cls, cases, shape):
        """
        Create a CCRVAM object instance from a list of cases.

        Parameters
        ----------
        cases : numpy.ndarray
            A 2D array where each row represents a case.
        shape : tuple
            Shape of the contingency table to create.

        Returns
        -------
        CCRVAM
            A new instance initialized with the probability matrix.

        Raises
        ------
        ValueError
            If the input cases are not 2-dimensional.
            If the shape tuple does not match the number of variables.

        Examples
        --------
        >>> cases = np.array([
            [0, 0, 2],
            [0, 1, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 2]
        ])
        >>> copula = GenericCCRVAM.from_cases(cases, (3, 2, 3))
        """
        if not isinstance(cases, np.ndarray):
            cases = np.array(cases)
            
        if cases.ndim != 2:
            raise ValueError("Cases must be a 2D array")
            
        if cases.shape[1] != len(shape):
            raise ValueError("Shape tuple must match number of variables")
            
        # Convert from 1-indexed input to 0-indexed categorical cases
        cases -= 1
        
        contingency_table = gen_case_form_to_contingency(cases, shape)
        return cls.from_contingency_table(contingency_table)
    
    def __init__(self, P):
        """Initialize with joint probability matrix P."""
        if not isinstance(P, np.ndarray):
            P = np.array(P)
            
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("P must contain values in [0,1]")
            
        if not np.allclose(P.sum(), 1.0):
            raise ValueError("P must sum to 1")
            
        self.P = P
        self.ndim = P.ndim
        
        # Calculate and store marginals for each axis
        self.marginal_pdfs = {}
        self.marginal_cdfs = {}
        self.scores = {}
        
        for axis in range(self.ndim):
            # Calculate marginal PDF
            pdf = P.sum(axis=tuple(i for i in range(self.ndim) if i != axis))
            self.marginal_pdfs[axis] = pdf
            
            # Calculate marginal CDF
            cdf = np.insert(np.cumsum(pdf), 0, 0)
            self.marginal_cdfs[axis] = cdf
            
            # Calculate scores
            self.scores[axis] = self._calculate_scores(cdf)
            
        # Store conditional PMFs
        self.conditional_pmfs = {}
        
    @property
    def contingency_table(self):
        """Get the contingency table by rescaling the probability matrix.
        
        This property converts the internal probability matrix (P) back to an 
        approximate contingency table of counts. Since the exact original counts
        cannot be recovered, it scales the probabilities by finding the smallest 
        non-zero probability and using its reciprocal as a multiplier.
        
        Returns
        -------
        numpy.ndarray
            A matrix of integer counts representing the contingency table.
            The values are rounded to the nearest integer after scaling.
        
        Notes
        -----
        The scaling process works by:
        1. Finding the smallest non-zero probability in the matrix
        2. Using its reciprocal as the scaling factor
        3. Multiplying all probabilities by this scale
        4. Rounding to nearest integers
        
        Warning
        -------
        This is an approximation of the original contingency table since the
        exact counts cannot be recovered from probabilities alone.
        """
        # Multiply by the smallest number that makes all entries close to integers
        scale = 1 / np.min(self.P[self.P > 0]) if np.any(self.P > 0) else 1
        return np.round(self.P * scale).astype(int)
        
    def calculate_CCRAM(self, predictors, response, scaled=False):
        """Calculate CCRAM with multiple conditioning axes.
        
        Parameters
        ----------
        predictors : list
            List of 1-indexed predictors axes for directional association
        response : int
            1-indexed target response axis for directional association
        scaled : bool, optional
            Whether to return standardized measure (default: False)
        """
        if not isinstance(predictors, (list, tuple)):
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
        axis_names: dict = None
    ) -> pd.DataFrame:
        """Get category predictions with multiple conditioning axes.
        
        Parameters
        ----------
        predictors : list
            List of 1-indexed predictors axes for category prediction
        response : int
            1-indexed target response axis for category prediction
        axis_names : dict, optional
            Dictionary mapping 1-indexed axis indices to names (default: None)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing source and predicted categories
        
        Examples
        --------
        >>> copula.get_predictions_ccr([1, 2], 3)
        
        Notes
        -----
        The DataFrame contains columns for each source axis category and the 
        predicted target axis category. The categories are 1-indexed.
        """
        if axis_names is None:
            axis_names = {i+1: f"X{i+1}" for i in range(self.ndim)}
        
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
            result[f'{axis_names[axis+1]} Category'] = cats + 1
        result[f'Predicted {axis_names[parsed_response+1]} Category'] = predictions + 1
        
        return result
    
    def calculate_ccs(self, var_index):
        """Calculate checkerboard scores for the specified variable index.
        
        Parameters
        ----------
        var_index : int
            1-Indexed axis of the variable for which to calculate scores
            
        Returns
        -------
        numpy.ndarray
            Array containing checkerboard scores for the given axis
        """
        parsed_axis = var_index - 1
        return self.scores[parsed_axis]
    
    def calculate_variance_ccs(self, var_index):
        """Calculate the variance of score S for the specified variable index.
        
        Parameters
        ----------
        var_index : int
            1-Indexed axis of the variable for which to calculate variance
            
        Returns
        -------
        float
            Variance of score S for the given axis
        """
        parsed_axis = var_index - 1
        return self._calculate_sigma_sq_S_vectorized(parsed_axis)
    
    def _calculate_conditional_pmf(self, target_axis, given_axes):
        """Helper Function: Calculate conditional PMF P(target|given)."""
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
        """Vectorized regression calculation for multiple conditioning axes."""
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
        """Helper Function: Calculate checkerboard scores from marginal CDF."""
        return [(marginal_cdf[j-1] + marginal_cdf[j])/2 
                for j in range(1, len(marginal_cdf))]
    
    def _lambda_function(self, u, ul, uj):
        """Helper Function: Calculate lambda function for checkerboard copula."""
        if u <= ul:
            return 0.0
        elif u >= uj:
            return 1.0
        else:
            return (u - ul) / (uj - ul)
        
    def _get_predicted_category(self, regression_value, marginal_cdf):
        """Helper Function: Get predicted category based on regression value."""
        return np.searchsorted(marginal_cdf[1:-1], regression_value, side='left')

    def _get_predicted_category_batched(self, regression_values, marginal_cdf):
        """Helper Function: Get predicted categories for multiple regression values."""
        return np.searchsorted(marginal_cdf[1:-1], regression_values, side='left')
    
    def _calculate_sigma_sq_S(self, axis):
        """Helper Function: Calculate variance of score S for given axis."""
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Calculate each term in the sum
        terms = []
        for i in range(len(self.marginal_pdfs[axis])):
            if i < len(u_prev) and i < len(u_next):
                term = u_prev[i] * u_next[i] * self.marginal_pdfs[axis][i]
                terms.append(term)
        
        # Calculate sigma_sq_S
        sigma_sq_S = sum(terms) / 4.0
        return sigma_sq_S

    def _calculate_sigma_sq_S_vectorized(self, axis):
        """Helper Function: Calculate variance of score S using vectorized operations."""
        # Get consecutive CDF values
        u_prev = self.marginal_cdfs[axis][:-1]
        u_next = self.marginal_cdfs[axis][1:]
        
        # Vectorized multiplication of all terms
        terms = u_prev * u_next * self.marginal_pdfs[axis]
        
        # Calculate sigma_sq_S
        sigma_sq_S = np.sum(terms) / 4.0
        return sigma_sq_S
    
    def _predict_category(self, source_category, predictors, response):
        """Helper Function: Predict category for target axis given source category."""
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
        return self._get_predicted_category(u_target, self.marginal_cdfs[response])
    
    def _predict_category_batched_multi(
        self, 
        source_categories, 
        predictors, 
        response
    ):
        if not isinstance(predictors, (list, tuple)):
            predictors = [predictors]
        
        """Vectorized prediction with multiple conditioning axes."""
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
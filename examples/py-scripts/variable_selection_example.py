#!/usr/bin/env python3
"""
Variable Selection using CCRVAM Package
=======================================

This script demonstrates how to perform all-possible-subset and best-subset 
variable selection using the ccrvam package. It includes:

1. All-possible-subset variable selection
2. Best-subset variable selection (with k predictors)
3. Bootstrap confidence intervals for selected models
4. Permutation test p-values for selected models
5. Visualization and reporting of results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, Tuple, Optional
import time
import warnings

# Import ccrvam functions - direct imports to avoid package initialization issues
from ccrvam.checkerboard.genccrvam import GenericCCRVAM
from ccrvam.checkerboard.genstatsim import (
    bootstrap_ccram, 
    permutation_test_ccram,
    bootstrap_predict_ccr_summary
)

warnings.filterwarnings("ignore")

class CCRAMVariableSelector:
    """
    A class for performing variable selection using CCRAM values.
    """
    
    def __init__(self, contingency_table: np.ndarray, variable_names: Optional[Dict[int, str]] = None):
        """
        Initialize the variable selector.
        
        Parameters
        ----------
        contingency_table : np.ndarray
            The contingency table for analysis
        variable_names : dict, optional
            Dictionary mapping 1-indexed variable numbers to names
        """
        self.table = contingency_table
        self.ccrvam = GenericCCRVAM.from_contingency_table(contingency_table)
        self.n_variables = contingency_table.ndim
        
        # Default variable names if not provided
        if variable_names is None:
            self.variable_names = {i+1: f"X{i+1}" for i in range(self.n_variables)}
        else:
            self.variable_names = variable_names
            
        self.results = {}
        
    def all_possible_subsets_selection(
        self, 
        response_var: int,
        min_predictors: int = 1,
        max_predictors: Optional[int] = None,
        scaled: bool = False,
        include_bootstrap: bool = True,
        include_permutation: bool = True,
        n_resamples: int = 999,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Perform all-possible-subset variable selection.
        
        Parameters
        ----------
        response_var : int
            1-indexed response variable
        min_predictors : int
            Minimum number of predictors to consider
        max_predictors : int, optional
            Maximum number of predictors to consider
        scaled : bool
            Whether to use scaled CCRAM (SCCRAM)
        include_bootstrap : bool
            Whether to include bootstrap confidence intervals
        include_permutation : bool
            Whether to include permutation test p-values
        n_resamples : int
            Number of bootstrap/permutation resamples
        confidence_level : float
            Confidence level for bootstrap intervals
        random_state : int, optional
            Random state for reproducibility
        parallel : bool
            Whether to use parallel processing
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with CCRAM values and statistics
        """
        print("Performing all-possible-subset variable selection...")
        print(f"Response variable: {self.variable_names[response_var]} (X{response_var})")
        
        # Get available predictor variables (exclude response)
        predictor_vars = [i for i in range(1, self.n_variables + 1) if i != response_var]
        
        if max_predictors is None:
            max_predictors = len(predictor_vars)
        else:
            max_predictors = min(max_predictors, len(predictor_vars))
            
        results = []
        total_combinations = sum(len(list(combinations(predictor_vars, r))) 
                               for r in range(min_predictors, max_predictors + 1))
        
        print(f"Testing {total_combinations} predictor combinations...")
        
        combination_count = 0
        
        # Iterate through all possible subset sizes
        for n_preds in range(min_predictors, max_predictors + 1):
            print(f"\nTesting combinations with {n_preds} predictor(s)...")
            
            # Generate all combinations of n_preds predictors
            for pred_combo in combinations(predictor_vars, n_preds):
                combination_count += 1
                
                # Calculate CCRAM
                start_time = time.time()
                ccram_value = self.ccrvam.calculate_CCRAM(
                    predictors=list(pred_combo),
                    response=response_var,
                    scaled=scaled
                )
                ccram_time = time.time() - start_time
                
                # Prepare result dictionary
                result = {
                    'combination_id': combination_count,
                    'n_predictors': n_preds,
                    'predictors': pred_combo,
                    'predictor_names': [self.variable_names[p] for p in pred_combo],
                    'ccram_value': ccram_value,
                    'ccram_computation_time': ccram_time
                }
                
                # Bootstrap confidence interval
                if include_bootstrap:
                    print(f"  [{combination_count}/{total_combinations}] Bootstrap for {pred_combo}...")
                    try:
                        bootstrap_result = bootstrap_ccram(
                            self.table,
                            predictors=list(pred_combo),
                            response=response_var,
                            scaled=scaled,
                            n_resamples=n_resamples,
                            confidence_level=confidence_level,
                            random_state=random_state,
                            parallel=parallel
                        )
                        result.update({
                            'bootstrap_ci_lower': bootstrap_result.confidence_interval[0],
                            'bootstrap_ci_upper': bootstrap_result.confidence_interval[1],
                            'bootstrap_se': bootstrap_result.standard_error
                        })
                    except Exception as e:
                        print(f"    Bootstrap failed: {e}")
                        result.update({
                            'bootstrap_ci_lower': np.nan,
                            'bootstrap_ci_upper': np.nan,
                            'bootstrap_se': np.nan
                        })
                
                # Permutation test
                if include_permutation:
                    print(f"  [{combination_count}/{total_combinations}] Permutation test for {pred_combo}...")
                    try:
                        perm_result = permutation_test_ccram(
                            self.table,
                            predictors=list(pred_combo),
                            response=response_var,
                            scaled=scaled,
                            alternative='greater',
                            n_resamples=n_resamples,
                            random_state=random_state,
                            parallel=parallel
                        )
                        result['permutation_pvalue'] = perm_result.p_value
                    except Exception as e:
                        print(f"    Permutation test failed: {e}")
                        result['permutation_pvalue'] = np.nan
                
                results.append(result)
                
                print(f"  [{combination_count}/{total_combinations}] {pred_combo}: CCRAM = {ccram_value:.6f}")
        
        # Convert to DataFrame and sort by CCRAM value
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ccram_value', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Store results
        self.results['all_subsets'] = results_df
        
        print("\nAll-possible-subset selection completed!")
        print(f"Best combination: {results_df.iloc[0]['predictor_names']} (CCRAM = {results_df.iloc[0]['ccram_value']:.6f})")
        
        return results_df
    
    def best_k_subset_selection(
        self,
        response_var: int,
        k: int,
        scaled: bool = False,
        include_bootstrap: bool = True,
        include_permutation: bool = True,
        n_resamples: int = 999,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Perform best-k-subset variable selection.
        
        Parameters
        ----------
        response_var : int
            1-indexed response variable
        k : int
            Number of predictors to select
        scaled : bool
            Whether to use scaled CCRAM (SCCRAM)
        include_bootstrap : bool
            Whether to include bootstrap confidence intervals
        include_permutation : bool
            Whether to include permutation test p-values
        n_resamples : int
            Number of bootstrap/permutation resamples
        confidence_level : float
            Confidence level for bootstrap intervals
        random_state : int, optional
            Random state for reproducibility
        parallel : bool
            Whether to use parallel processing
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with CCRAM values and statistics for k-predictor models
        """
        print("Performing best-k-subset variable selection...")
        print(f"Response variable: {self.variable_names[response_var]} (X{response_var})")
        
        # Get available predictor variables (exclude response)
        predictor_vars = [i for i in range(1, self.n_variables + 1) if i != response_var]
        
        if k > len(predictor_vars):
            raise ValueError(f"k={k} is larger than available predictors ({len(predictor_vars)})")
        
        # Generate all combinations of k predictors
        k_combinations = list(combinations(predictor_vars, k))
        print(f"Testing {len(k_combinations)} combinations of {k} predictors...")
        
        results = []
        
        for i, pred_combo in enumerate(k_combinations, 1):
            # Calculate CCRAM
            start_time = time.time()
            ccram_value = self.ccrvam.calculate_CCRAM(
                predictors=list(pred_combo),
                response=response_var,
                scaled=scaled
            )
            ccram_time = time.time() - start_time
            
            # Prepare result dictionary
            result = {
                'combination_id': i,
                'predictors': pred_combo,
                'predictor_names': [self.variable_names[p] for p in pred_combo],
                'ccram_value': ccram_value,
                'ccram_computation_time': ccram_time
            }
            
            # Bootstrap confidence interval
            if include_bootstrap:
                print(f"  [{i}/{len(k_combinations)}] Bootstrap for {pred_combo}...")
                try:
                    bootstrap_result = bootstrap_ccram(
                        self.table,
                        predictors=list(pred_combo),
                        response=response_var,
                        scaled=scaled,
                        n_resamples=n_resamples,
                        confidence_level=confidence_level,
                        random_state=random_state,
                        parallel=parallel
                    )
                    result.update({
                        'bootstrap_ci_lower': bootstrap_result.confidence_interval[0],
                        'bootstrap_ci_upper': bootstrap_result.confidence_interval[1],
                        'bootstrap_se': bootstrap_result.standard_error
                    })
                except Exception as e:
                    print(f"    Bootstrap failed: {e}")
                    result.update({
                        'bootstrap_ci_lower': np.nan,
                        'bootstrap_ci_upper': np.nan,
                        'bootstrap_se': np.nan
                    })
            
            # Permutation test
            if include_permutation:
                print(f"  [{i}/{len(k_combinations)}] Permutation test for {pred_combo}...")
                try:
                    perm_result = permutation_test_ccram(
                        self.table,
                        predictors=list(pred_combo),
                        response=response_var,
                        scaled=scaled,
                        alternative='greater',
                        n_resamples=n_resamples,
                        random_state=random_state,
                        parallel=parallel
                    )
                    result['permutation_pvalue'] = perm_result.p_value
                except Exception as e:
                    print(f"    Permutation test failed: {e}")
                    result['permutation_pvalue'] = np.nan
            
            results.append(result)
            
            print(f"  [{i}/{len(k_combinations)}] {pred_combo}: CCRAM = {ccram_value:.6f}")
        
        # Convert to DataFrame and sort by CCRAM value
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ccram_value', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Store results
        self.results[f'best_{k}_subsets'] = results_df
        
        print(f"\nBest-{k}-subset selection completed!")
        print(f"Best combination: {results_df.iloc[0]['predictor_names']} (CCRAM = {results_df.iloc[0]['ccram_value']:.6f})")
        
        return results_df
    
    def plot_selection_results(
        self,
        results_df: pd.DataFrame,
        title: str = "Variable Selection Results",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot variable selection results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from variable selection
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: CCRAM values by rank
        axes[0, 0].plot(results_df['rank'], results_df['ccram_value'], 'bo-')
        axes[0, 0].set_xlabel('Rank')
        axes[0, 0].set_ylabel('CCRAM Value')
        axes[0, 0].set_title('CCRAM Values by Rank')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Bootstrap confidence intervals (if available)
        if 'bootstrap_ci_lower' in results_df.columns:
            top_10 = results_df.head(10)
            # Filter out rows with NaN values in bootstrap data
            valid_bootstrap = top_10.dropna(subset=['bootstrap_ci_lower', 'bootstrap_ci_upper'])
            if len(valid_bootstrap) > 0:
                x_pos = range(len(valid_bootstrap))
                # Calculate error values, ensuring they're positive
                lower_err = np.maximum(0, valid_bootstrap['ccram_value'] - valid_bootstrap['bootstrap_ci_lower'])
                upper_err = np.maximum(0, valid_bootstrap['bootstrap_ci_upper'] - valid_bootstrap['ccram_value'])
                
                axes[0, 1].errorbar(
                    x_pos, 
                    valid_bootstrap['ccram_value'],
                    yerr=[lower_err, upper_err],
                    fmt='o', capsize=5
                )
                axes[0, 1].set_xlabel('Top Models')
                axes[0, 1].set_ylabel('CCRAM Value')
                axes[0, 1].set_title(f'Bootstrap Confidence Intervals (Top {len(valid_bootstrap)})')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Valid Bootstrap Data', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Bootstrap Confidence Intervals')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Bootstrap Data', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Bootstrap Confidence Intervals')
        
        # Plot 3: P-values (if available)
        if 'permutation_pvalue' in results_df.columns:
            valid_pvals = results_df.dropna(subset=['permutation_pvalue'])
            if len(valid_pvals) > 0:
                axes[1, 0].scatter(valid_pvals['ccram_value'], valid_pvals['permutation_pvalue'], alpha=0.6)
                axes[1, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α = 0.05')
                axes[1, 0].set_xlabel('CCRAM Value')
                axes[1, 0].set_ylabel('Permutation P-value')
                axes[1, 0].set_title('P-values vs CCRAM Values')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Valid P-value Data', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'No P-value Data', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Permutation P-values')
        
        # Plot 4: Number of predictors vs CCRAM (if applicable)
        if 'n_predictors' in results_df.columns:
            pred_summary = results_df.groupby('n_predictors')['ccram_value'].agg(['max', 'mean']).reset_index()
            axes[1, 1].plot(pred_summary['n_predictors'], pred_summary['max'], 'ro-', label='Best')
            axes[1, 1].plot(pred_summary['n_predictors'], pred_summary['mean'], 'bs-', label='Average')
            axes[1, 1].set_xlabel('Number of Predictors')
            axes[1, 1].set_ylabel('CCRAM Value')
            axes[1, 1].set_title('CCRAM vs Number of Predictors')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # For best-k selection, show distribution of CCRAM values
            axes[1, 1].hist(results_df['ccram_value'], bins=min(20, len(results_df)//2), alpha=0.7)
            axes[1, 1].set_xlabel('CCRAM Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of CCRAM Values')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def summarize_results(self, results_df: pd.DataFrame, top_n: int = 5) -> None:
        """
        Print a summary of variable selection results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from variable selection
        top_n : int
            Number of top models to display
        """
        print("\n" + "="*60)
        print("VARIABLE SELECTION RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nTotal models evaluated: {len(results_df)}")
        print(f"Best CCRAM value: {results_df.iloc[0]['ccram_value']:.6f}")
        print(f"Worst CCRAM value: {results_df.iloc[-1]['ccram_value']:.6f}")
        
        if 'n_predictors' in results_df.columns:
            pred_counts = results_df['n_predictors'].value_counts().sort_index()
            print("\nModels by number of predictors:")
            for n_pred, count in pred_counts.items():
                print(f"  {n_pred} predictors: {count} models")
        
        print(f"\nTop {top_n} models:")
        print("-"*60)
        
        for i in range(min(top_n, len(results_df))):
            row = results_df.iloc[i]
            print(f"Rank {i+1}: {row['predictor_names']}")
            print(f"  CCRAM: {row['ccram_value']:.6f}")
            
            if 'bootstrap_ci_lower' in row and not pd.isna(row['bootstrap_ci_lower']):
                print(f"  Bootstrap CI: [{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}]")
                print(f"  Bootstrap SE: {row['bootstrap_se']:.6f}")
            
            if 'permutation_pvalue' in row and not pd.isna(row['permutation_pvalue']):
                print(f"  P-value: {row['permutation_pvalue']:.6f}")
            
            print()
        
        print("="*60)


def create_example_data() -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Create example 4D contingency table for demonstration.
    
    Returns
    -------
    tuple
        Contingency table and variable names
    """
    # Create a 4D contingency table
    table = np.zeros((2, 3, 2, 6), dtype=int)
    
    # Fill with example data (from test fixtures)
    table[0, 0, 0, 1] = 1
    table[0, 0, 0, 4] = 2
    table[0, 0, 0, 5] = 4
    table[0, 0, 1, 3] = 1
    table[0, 0, 1, 4] = 3
    table[0, 1, 0, 1] = 2
    table[0, 1, 0, 2] = 3
    table[0, 1, 0, 4] = 6
    table[0, 1, 0, 5] = 4
    table[0, 1, 1, 1] = 1
    table[0, 1, 1, 3] = 2
    table[0, 1, 1, 5] = 1
    table[0, 2, 0, 4] = 2
    table[0, 2, 0, 5] = 2
    table[0, 2, 1, 2] = 1
    table[0, 2, 1, 3] = 1
    table[0, 2, 1, 4] = 3
    table[1, 0, 0, 2] = 3
    table[1, 0, 0, 4] = 1
    table[1, 0, 0, 5] = 2
    table[1, 0, 1, 1] = 1
    table[1, 0, 1, 4] = 3
    table[1, 1, 0, 1] = 3
    table[1, 1, 0, 2] = 4
    table[1, 1, 0, 3] = 5
    table[1, 1, 0, 4] = 6
    table[1, 1, 0, 5] = 2
    table[1, 1, 1, 0] = 1
    table[1, 1, 1, 1] = 4
    table[1, 1, 1, 2] = 4
    table[1, 1, 1, 3] = 3
    table[1, 1, 1, 5] = 1
    table[1, 2, 0, 0] = 2
    table[1, 2, 0, 1] = 2
    table[1, 2, 0, 2] = 1
    table[1, 2, 0, 3] = 5
    table[1, 2, 0, 4] = 2
    table[1, 2, 1, 0] = 2
    table[1, 2, 1, 2] = 2
    table[1, 2, 1, 3] = 3
    
    variable_names = {
        1: "Pelvic_Incidence",
        2: "Pelvic_Tilt", 
        3: "Lordosis_Angle",
        4: "Spine_Outcome"
    }
    
    return table, variable_names


def main():
    """
    Main function demonstrating variable selection workflows.
    """
    print("CCRVAM Variable Selection Example")
    print("=" * 50)
    
    # Create example data
    print("\n1. Loading example data...")
    table, var_names = create_example_data()
    print(f"Data shape: {table.shape}")
    print(f"Variables: {list(var_names.values())}")
    print(f"Total observations: {table.sum()}")
    
    # Initialize variable selector
    selector = CCRAMVariableSelector(table, var_names)
    
    # Example 1: Best-2-subset selection
    print("\n" + "="*50)
    print("EXAMPLE 1: Best-2-Subset Selection")
    print("="*50)
    
    best_2_results = selector.best_k_subset_selection(
        response_var=4,  # Spine_Outcome
        k=2,
        scaled=False,
        include_bootstrap=True,
        include_permutation=True,
        n_resamples=199,  # Reduced for demo speed
        random_state=42,
        parallel=True
    )
    
    # Summarize results
    selector.summarize_results(best_2_results, top_n=3)
    
    # Plot results
    selector.plot_selection_results(
        best_2_results, 
        title="Best-2-Subset Selection Results",
        save_path="best_2_subset_results.png"
    )
    
    # Example 2: All-possible-subset selection (limited for demo)
    print("\n" + "="*50)
    print("EXAMPLE 2: All-Possible-Subset Selection")
    print("="*50)
    
    all_subset_results = selector.all_possible_subsets_selection(
        response_var=4,  # Spine_Outcome
        min_predictors=1,
        max_predictors=2,  # Limited for demo speed
        scaled=False,
        include_bootstrap=True,
        include_permutation=True,
        n_resamples=199,  # Reduced for demo speed
        random_state=42,
        parallel=True
    )
    
    # Summarize results
    selector.summarize_results(all_subset_results, top_n=5)
    
    # Plot results
    selector.plot_selection_results(
        all_subset_results,
        title="All-Possible-Subset Selection Results",
        save_path="all_subset_results.png"
    )
    
    # Example 3: Compare scaled vs unscaled CCRAM
    print("\n" + "="*50)
    print("EXAMPLE 3: Scaled vs Unscaled CCRAM Comparison")
    print("="*50)
    
    print("Testing scaled CCRAM (SCCRAM)...")
    scaled_results = selector.best_k_subset_selection(
        response_var=4,
        k=2,
        scaled=True,  # Use SCCRAM
        include_bootstrap=True,
        include_permutation=False,  # Skip for speed
        n_resamples=99,
        random_state=42,
        parallel=True
    )
    
    # Compare top models
    print("\nComparison of top models:")
    print("CCRAM (unscaled):")
    print(f"  Best: {best_2_results.iloc[0]['predictor_names']} = {best_2_results.iloc[0]['ccram_value']:.6f}")
    print("SCCRAM (scaled):")
    print(f"  Best: {scaled_results.iloc[0]['predictor_names']} = {scaled_results.iloc[0]['ccram_value']:.6f}")
    
    # Example 4: Prediction summary for best model
    print("\n" + "="*50)
    print("EXAMPLE 4: Prediction Summary for Best Model")
    print("="*50)
    
    best_predictors = list(best_2_results.iloc[0]['predictors'])
    best_names = best_2_results.iloc[0]['predictor_names']
    
    print(f"Generating prediction summary for: {best_names}")
    
    # Generate prediction summary with bootstrap
    pred_summary = bootstrap_predict_ccr_summary(
        table=table,
        predictors=best_predictors,
        predictors_names=best_names,
        response=4,
        response_name=var_names[4],
        n_resamples=199,
        random_state=42,
        parallel=True
    )
    
    print("\nPrediction Summary (first 5 columns):")
    print(pred_summary.iloc[:, :5])
    
    # Plot prediction summary
    fig, ax = pred_summary.plot_predictions_summary(
        figsize=(12, 8),
        plot_type='heatmap',
        save_path="prediction_summary.png"
    )
    # Set the title after creation to avoid kwargs conflicts
    fig.suptitle(f"Prediction Summary: {' + '.join(best_names)} → {var_names[4]}", fontsize=14)
    plt.show()
    
    print("\n" + "="*50)
    print("VARIABLE SELECTION ANALYSIS COMPLETE")
    print("="*50)
    print("Generated files:")
    print("- best_2_subset_results.png")
    print("- all_subset_results.png") 
    print("- prediction_summary.png")
    print("\nFor larger datasets, consider:")
    print("- Increasing n_resamples for more precise estimates")
    print("- Using parallel=True for faster computation")
    print("- Limiting max_predictors for all-subset selection")
    print("- Using scaled=True for comparing models with different numbers of predictors")


if __name__ == "__main__":
    main() 
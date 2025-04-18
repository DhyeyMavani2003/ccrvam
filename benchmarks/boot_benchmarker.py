import time
import numpy as np
# import cupy as cp
from ccrvam import (
    bootstrap_ccram,
    permutation_test_ccram,
    bootstrap_predict_ccr_summary,
    save_predictions,
    DataProcessor
)

def time_operation(operation_name, operation_func, *args, **kwargs):
    start_time = time.time()
    result = operation_func(*args, **kwargs)
    end_time = time.time()
    print(f"\nTime taken for {operation_name}: {end_time - start_time:.4f} seconds")
    return result

def run_operations_sequentially(operations):
    """Run operations sequentially instead of in parallel"""
    results = []
    for op_name, op_func, args, kwargs in operations:
        results.append(time_operation(op_name, op_func, *args, **kwargs))
    return results

def main():
    try:
        # 2-Dimensional Case
        print("\n=== 2-Dimensional Case ===")
        
        # Create Sample Contingency Table
        contingency_table = np.array([
            [0, 0, 20],
            [0, 10, 0],
            [20, 0, 0],
            [0, 10, 0],
            [0, 0, 20]
        ])

        # Define operations for 2D case
        operations_2d = [
            ("Bootstrap CCRAM analysis", bootstrap_ccram,
             [contingency_table], {"predictors": [1], "response": 2, "n_resamples": 9999, "scaled": False, "confidence_level": 0.95, "method": "percentile"}),
            ("Bootstrap SCCRAM analysis", bootstrap_ccram,
             [contingency_table], {"predictors": [1], "response": 2, "n_resamples": 9999, "scaled": True, "confidence_level": 0.95, "method": "percentile"}),
            ("Bootstrap prediction summary", bootstrap_predict_ccr_summary,
             [contingency_table], {"predictors": [1], "predictors_names": ["X"], "response": 2, "response_name": "Y", "n_resamples": 1000}),
            ("Permutation test CCRAM", permutation_test_ccram,
             [contingency_table], {"predictors": [1], "response": 2, "scaled": False, "alternative": 'greater', "n_resamples": 9999}),
            ("Permutation test SCCRAM", permutation_test_ccram,
             [contingency_table], {"predictors": [1], "response": 2, "scaled": True, "alternative": 'greater', "n_resamples": 9999})
        ]

        # Run 2D operations sequentially
        print("\n--- Running 2D Operations ---")
        ccram_result, sccram_result, prediction_matrix, perm_result, scaled_perm_result = run_operations_sequentially(operations_2d)

        # Print 2D results
        print("\n--- Bootstrap CCRAM Analysis ---")
        print(f"Metric Name: {ccram_result.metric_name}")
        print(f"Observed Value: {ccram_result.observed_value:.4f}")
        print(f"95% CI: ({ccram_result.confidence_interval[0]:.4f}, {ccram_result.confidence_interval[1]:.4f})")
        print(f"Standard Error: {ccram_result.standard_error:.4f}")

        print("\n--- Bootstrap SCCRAM Analysis ---")
        print(f"Metric Name: {sccram_result.metric_name}")
        print(f"Observed Value: {sccram_result.observed_value:.4f}")
        print(f"95% CI: ({sccram_result.confidence_interval[0]:.4f}, {sccram_result.confidence_interval[1]:.4f})")
        print(f"Standard Error: {sccram_result.standard_error:.4f}")

        print("\n--- Bootstrap Prediction Summary ---")
        print("\nPrediction Matrix:")
        print(prediction_matrix)

        print("\n--- Permutation Test CCRAM ---")
        print(f"Metric Name: {perm_result.metric_name}")
        print(f"Observed Value: {perm_result.observed_value:.4f}")
        print(f"P-Value: {perm_result.p_value:.4f}")

        print("\n--- Permutation Test SCCRAM ---")
        print(f"Metric Name: {scaled_perm_result.metric_name}")
        print(f"Observed Value: {scaled_perm_result.observed_value:.4f}")
        print(f"P-Value: {scaled_perm_result.p_value:.4f}")

        # 4-Dimensional Case
        print("\n=== 4-Dimensional Case ===")
        
        # Define variables and dimensions
        var_list_4d = ["x1", "x2", "x3", "pain"]
        category_map_4d = {
            "pain": {
                "worse": 1,
                "same": 2,
                "slight.improvement": 3,
                "moderate.improvement": 4,
                "marked.improvement": 5,
                "complete.relief": 6
            },
        }
        data_dimension = (2, 3, 2, 6)

        # Load data from case form
        print("\n--- Loading Data ---")
        rda_contingency_table_np = time_operation("Loading case form data", 
                                                DataProcessor.load_data,
                                                "./data/caseform.pain.txt",
                                                data_form="case_form",
                                                dimension=data_dimension,
                                                var_list=var_list_4d,
                                                category_map=category_map_4d,
                                                named=True,
                                                delimiter="\t")
        rda_contingency_table = rda_contingency_table_np

        # Define operations for 4D case
        operations_4d = [
            ("Bootstrap CCRAM analysis (4D)", bootstrap_ccram,
             [rda_contingency_table], {"predictors": [1, 2, 3], "response": 4, "confidence_level": 0.95, "scaled": False, "method": "percentile", "n_resamples": 9999, "random_state": 8990}),
            ("Bootstrap SCCRAM analysis (4D)", bootstrap_ccram,
             [rda_contingency_table], {"predictors": [1, 2, 3], "response": 4, "confidence_level": 0.95, "scaled": True, "method": "BCa", "n_resamples": 9999, "random_state": 8990}),
            ("Bootstrap prediction summary (4D)", bootstrap_predict_ccr_summary,
             [rda_contingency_table], {"predictors": [1, 2, 3], "predictors_names": ["X1", "X2", "X3"], "response": 4, "response_name": "Pain", "n_resamples": 9999, "random_state": 8990, "parallel": True}),
            ("Permutation test CCRAM (4D)", permutation_test_ccram,
             [rda_contingency_table], {"predictors": [1, 2, 3], "response": 4, "scaled": False, "alternative": 'greater', "n_resamples": 9999, "random_state": 8990}),
            ("Permutation test SCCRAM (4D)", permutation_test_ccram,
             [rda_contingency_table], {"predictors": [1, 2, 3], "response": 4, "scaled": True, "alternative": 'greater', "n_resamples": 9999, "random_state": 8990})
        ]

        # Run 4D operations sequentially
        print("\n--- Running 4D Operations ---")
        rda_ccram_result, rda_sccram_result, rda_prediction_matrix, rda_perm_result, rda_scaled_perm_result = run_operations_sequentially(operations_4d)

        # Print 4D results
        print("\n--- Bootstrap CCRAM Analysis (4D) ---")
        print(f"Metric Name: {rda_ccram_result.metric_name}")
        print(f"Observed Value: {rda_ccram_result.observed_value:.4f}")
        print(f"95% CI: ({rda_ccram_result.confidence_interval[0]:.4f}, {rda_ccram_result.confidence_interval[1]:.4f})")
        print(f"Standard Error: {rda_ccram_result.standard_error:.4f}")

        print("\n--- Bootstrap SCCRAM Analysis (4D) ---")
        print(f"Metric Name: {rda_sccram_result.metric_name}")
        print(f"Observed Value: {rda_sccram_result.observed_value:.4f}")
        print(f"95% CI: ({rda_sccram_result.confidence_interval[0]:.4f}, {rda_sccram_result.confidence_interval[1]:.4f})")
        print(f"Standard Error: {rda_sccram_result.standard_error:.4f}")

        print("\n--- Bootstrap Prediction Summary (4D) ---")
        print("\nPrediction Matrix:")
        print(rda_prediction_matrix)

        # Save predictions
        print("\n--- Saving Predictions ---")
        time_operation("Saving predictions", 
                      save_predictions,
                      prediction_matrix=rda_prediction_matrix, 
                      save_path="generated_pred_data/rda_prediction_matrix.csv", 
                      format="csv")

        print("\n--- Permutation Test CCRAM (4D) ---")
        print(f"Metric Name: {rda_perm_result.metric_name}")
        print(f"Observed Value: {rda_perm_result.observed_value:.4f}")
        print(f"P-Value: {rda_perm_result.p_value:.4f}")

        print("\n--- Permutation Test SCCRAM (4D) ---")
        print(f"Metric Name: {rda_scaled_perm_result.metric_name}")
        print(f"Observed Value: {rda_scaled_perm_result.observed_value:.4f}")
        print(f"P-Value: {rda_scaled_perm_result.p_value:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
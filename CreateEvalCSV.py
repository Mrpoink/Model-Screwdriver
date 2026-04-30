import json
import pandas as pd
import numpy as np
import ast
import csv
import pathlib
from scipy import stats

def aggregate_evals(root_dir, output_file="eval_summary.csv"):
    results = []
    all_keys = set()
    
    path = pathlib.Path(root_dir)
    # Recursively find all json files in your Model-Screwdriver directory
    files = list(path.rglob("*.json"))
    
    # Sort by folder directory then filename as requested
    files.sort(key=lambda p: (p.parent, p.name))
    
    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                
                # Unpack configuration and metrics
                row = {
                    "directory": str(file_path.parent),
                    "filename": file_path.name,
                    "timestamp": data.get("timestamp"),
                    **data.get("configuration", {}),
                    **data.get("metrics", {})
                }
                results.append(row)
                # Track every unique key found across all files
                all_keys.update(row.keys())
            except (json.JSONDecodeError, KeyError):
                continue

    if not results:
        print("No valid evaluation files found.")
        return

    # Sort keys so 'directory', 'filename', and 'timestamp' come first for readability
    fixed_headers = ["directory", "filename", "timestamp"]
    remaining_headers = sorted(list(all_keys - set(fixed_headers)))
    headers = fixed_headers + remaining_headers

    with open(output_file, 'w', newline='') as f:
        # extrasaction='ignore' prevents crashes if a row somehow has a mystery key
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Successfully processed {len(results)} files into {output_file}")
    
def calculate_statistical_significance(base_scores, steered_scores, alpha=0.05):
    """
    Calculates mean, std, t-statistic, and p-value for paired base vs steered scores.
    """
    if len(base_scores) < 2 or len(steered_scores) < 2:
        return {"error": "Not enough data points for a t-test (needs >= 2)."}
    
    # Calculate differences (improvements)
    differences = np.array(steered_scores) - np.array(base_scores)
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
    
    # Paired t-test
    # If all differences are exactly 0, ttest_rel returns NaN for t-stat and p-value
    if np.all(differences == 0):
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_rel(steered_scores, base_scores)
    
    # Determine if overall positive/negative and significant
    is_significant = bool(p_value < alpha)
    direction = "positive" if mean_diff > 0 else "negative" if mean_diff < 0 else "neutral"
    
    overall_result = "flat"
    if is_significant:
        overall_result = "significantly positive" if direction == "positive" else "significantly negative"
    else:
        overall_result = f"insignificantly {direction}"
        
    return {
        "mean_improvement": float(mean_diff),
        "std_improvement": float(std_diff),
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "is_significant": is_significant,
        "overall_result": overall_result
    }
    
def analyze_eval_logs(csv_file_path, output_json_path):
    # Load the data
    df = pd.read_csv(csv_file_path)
    df['version'] = df['filename'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) and '-' in str(x) else 'unknown')
    
    eval_columns = [col for col in df.columns if 'end stats' in col]
    final_results = {}

    for version, group in df.groupby('version'):
        group = group.sort_values(by='iteration')
        
        version_data = {
            "internal_parameters": [],
            "benchmark_statistics": {},
            "rank_statistics": {}
        }

        # Track parameters and gather raw scores per rank
        rank_raw_scores = {}
        
        for _, row in group.iterrows():
            loop_num = int(row['iteration'])
            try:
                params = ast.literal_eval(row['internal_params'])
                version_data["internal_parameters"].append({
                    "loop": loop_num,
                    "beta": params.get("beta"),
                    "restore_mag": params.get("restore_mag")
                })
            except Exception:
                pass
                
            rank_raw_scores[loop_num] = {"base": [], "steered": []}

        # Gather data for Benchmark-level statistics (Across all loops)
        benchmark_tracking = {col: {"base_acc": [], "steered_acc": [], "base_f1": [], "steered_f1": [], "base_ari": [], "steered_ari": []} for col in eval_columns}
        
        for _, row in group.iterrows():
            loop_num = int(row['iteration'])
            
            for col in eval_columns:
                if pd.notna(row[col]):
                    try:
                        stats_dict = ast.literal_eval(row[col])
                        
                        # Gather for benchmark level
                        benchmark_tracking[col]["base_acc"].append(stats_dict.get('base_accuracy', 0.0))
                        benchmark_tracking[col]["steered_acc"].append(stats_dict.get('steered_accuracy', 0.0))
                        benchmark_tracking[col]["base_f1"].append(stats_dict.get('base_f1', 0.0))
                        benchmark_tracking[col]["steered_f1"].append(stats_dict.get('steered_f1', 0.0))
                        benchmark_tracking[col]["base_ari"].append(stats_dict.get('base_ari', 0.0))
                        benchmark_tracking[col]["steered_ari"].append(stats_dict.get('steered_ari', 0.0))
                        
                        # Gather for rank level (aggregating F1 as a proxy for overall performance)
                        rank_raw_scores[loop_num]["base"].append(stats_dict.get('base_f1', 0.0))
                        rank_raw_scores[loop_num]["steered"].append(stats_dict.get('steered_f1', 0.0))
                        
                    except Exception:
                        continue

        # 1. Calculate Benchmark Statistics (Across all loops for this model)
        for col, metrics in benchmark_tracking.items():
            dataset_name = col.replace(' end stats', '')
            
            version_data["benchmark_statistics"][dataset_name] = {
                "accuracy": calculate_statistical_significance(metrics["base_acc"], metrics["steered_acc"]),
                "f1_score": calculate_statistical_significance(metrics["base_f1"], metrics["steered_f1"]),
                "ari_score": calculate_statistical_significance(metrics["base_ari"], metrics["steered_ari"])
            }

        # 2. Calculate Rank Statistics (Across all benchmarks for a specific loop)
        for loop_num, scores in rank_raw_scores.items():
            version_data["rank_statistics"][f"Loop_{loop_num}"] = calculate_statistical_significance(
                scores["base"], 
                scores["steered"]
            )

        final_results[f"version_{version}"] = version_data

    with open(output_json_path, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)
        
    print(f"Statistical analysis complete. Results successfully saved to {output_json_path}")
    return final_results

if __name__ == "__main__":
    # Pointed to your logs directory based on your traceback
    aggregate_evals(root_dir="./eval_logs")
    try:
        results = analyze_eval_logs(csv_file_path="eval_summary.csv", output_json_path="final_results.json")
    except FileNotFoundError:
        print(f"Error: Could not find 'eval_summary.csv'. Please ensure the file is in the same directory.")
import os
import json
import torch
from datetime import datetime
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import silhouette_score
from transformers import BertModel, BertTokenizer

from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as BuildDatasetMain
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.Tools import inject_weights, remove_weights
from ScrewDriver.ScrewDriverTrain import main as ScrewdriverTrainMain

def log_evaluation(eval_task_config, metrics, iteration, pipeline_name, log_dir="eval_logs"):
    """Saves the highly detailed JSON telemetry for a specific evaluation run."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"loop{iteration}_Train-{pipeline_name}_Eval-{eval_task_config['task_name']}_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "loop_iteration": iteration,
        "training_pipeline": pipeline_name,
        "evaluation_configuration": eval_task_config,
        "metrics": metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4)

def evaluate_model(eval_task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, iteration, pipeline_name, eval_samples=500):
    print(f"      Running {eval_samples}-Sample ZERO-SHOT Clustering on {eval_task_config['task_name']}...")
    
    base_features = list()
    steered_features = list()
    active_layers_snapshot = list()
    
    # 1. LOAD UNSEEN DATA & LABELS
    if eval_task_config['config_name']:
        dataset = load_dataset(eval_task_config['dataset_path'], eval_task_config['config_name'], split=eval_task_config['split'])
    else:
        dataset = load_dataset(eval_task_config['dataset_path'], split=eval_task_config['split'])

    # Safely extract text and align it with ground-truth labels
    valid_data = [(row.get('sentence', row.get('text')), row['label']) for row in dataset]
    # Filter for length and sample size
    valid_data = [(t, l) for t, l in valid_data if len(t.split()) < 100][:eval_samples]
    
    texts = [item[0] for item in valid_data]
    labels = [item[1] for item in valid_data]

    prompt_emb = harvester.embed_prompt(eval_task_config['task_label']).unsqueeze(0).to(device)

    # 2. INFERENCE LOOP
    for idx, text in enumerate(texts):
        inputs_base = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        # A. Base State
        with torch.inference_mode():
            base_features.append(large_model(**inputs_base).pooler_output.cpu().numpy())

        # B. Generate Weights
        A_small_b, B_small_b, _ = harvester.extract_task_matrices(
            small_model, [eval_task_config['baseline_prompt'] + " " + text], 
            [eval_task_config['task_label'] + " " + text], is_small=True, calc_variance=False
        )
        
        with torch.inference_mode():
            A_large, B_large, gate = screwdriver(
                A_small_b[0].unsqueeze(0).to(device), 
                B_small_b[0].unsqueeze(0).to(device), 
                prompt_emb, 
                hard=False 
            )
            # TOP-K ROUTING: Force the top 3 layers to open
            active_layers = torch.topk(gate.squeeze(), k=3).indices.tolist()
            active_layers_snapshot = active_layers 
            delta_W_seq = torch.matmul(B_large.squeeze(0), A_large.squeeze(0)).transpose(-1, -2)

        # C. Inject Weights
        injected_weights = []
        for l_idx in active_layers:
            scaled_W = delta_W_seq[l_idx] * 0.5 # Added a 0.5 dampener to prevent overshooting
            inject_weights(large_model, l_idx, scaled_W)
            injected_weights.append((l_idx, scaled_W))

        # D. Steered State
        with torch.inference_mode():
            steered_features.append(large_model(**inputs_base).pooler_output.cpu().numpy())

        # Reset Weights
        for l_idx, w in injected_weights:
            remove_weights(large_model, l_idx, w)

    # 3. UNSUPERVISED CLUSTERING METRIC (Silhouette Score)
    X_base = np.vstack(base_features)
    X_steered = np.vstack(steered_features)

    if np.isnan(X_base).any() or np.isnan(X_steered).any():
        X_base, X_steered = np.nan_to_num(X_base), np.nan_to_num(X_steered)

    # Calculate how tightly the data naturally clusters based on the labels
    base_silhouette = silhouette_score(X_base, labels)
    steered_silhouette = silhouette_score(X_steered, labels)
    
    # Positive means the Screwdriver successfully pulled the task concepts apart!
    cluster_improvement = steered_silhouette - base_silhouette

    metrics = {
        "base_cluster_score": float(base_silhouette),
        "steered_cluster_score": float(steered_silhouette),
        "cluster_improvement": float(cluster_improvement),
        "average_layers_altered": len(active_layers_snapshot)
    }

    log_evaluation(eval_task_config, metrics, iteration, pipeline_name)
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==========================================
    # ZERO-SHOT EVALUATION BENCHMARKS
    # ==========================================
    # These are the "Cousin" tasks. The Screwdriver never sees these during training.
    EVAL_CONFIGS = {
        "finance_sentiment": {
            "dataset_path": "FinanceMTEB/financial_phrasebank", "config_name": "default", "split": "train",
            "task_label": "Analyze the sentiment of this text:",
            "baseline_prompt": "The event occurred on a Tuesday afternoon."
        },
        "tweet_emotion": {
            "dataset_path": "tweet_eval", "config_name": "emotion", "split": "validation",
            "task_label": "Determine the emotion of this text:",
            "baseline_prompt": "The event occurred on a Tuesday afternoon."
        }
    }

    # Your existing training pipelines (We do not touch the master data)
    PIPELINES = ["imdb_sentiment", "ag_news_classification", "glue_sst2", "combined"]
    
    eval_samples = 500

    # ==========================================
    # THE MASTER GAUNTLET
    # ==========================================
    for iteration in range(1, 21):
        print("\n" + "#"*60)
        print(f"      STARTING MASTER LOOP {iteration} OF 20")
        print("#"*60)
        
        for pipeline_name in PIPELINES:
            print(f"\n{'='*40}\nEXECUTING FULL PIPELINE: {pipeline_name.upper()}\n{'='*40}")
            
            #--- 1. BUILD DATASET (Uses your existing extraction logic) ---
            # print(f"  [1/3] Building Dataset for {pipeline_name}...")
            # BuildDatasetMain()
            
            #--- 2. TRAIN SCREWDRIVER ---
            print(f"\n  [2/3] Training Screwdriver for {pipeline_name}...")
            ScrewdriverTrainMain(task_name=pipeline_name, task_label=pipeline_name)
            
            # --- 3. ZERO-SHOT EVALUATION ---
            print(f"\n  [3/3] Evaluating Screwdriver for {pipeline_name} on Unseen Cousins...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
            large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
            harvester = Harvester(small_model, large_model, tokenizer, device=device)
            
            screwdriver = ModelScrewDriver(d_small=768, d_large=1024, d_prompt=768, target_rank=12, num_small_layers=12, num_large_layers=24).to(device)
            weights_path = f"ModelScrewdriver_{pipeline_name}.pth"
            
            # Catch file not found if training failed
            if os.path.exists(weights_path):
                screwdriver.load_state_dict(torch.load(weights_path, weights_only=True))
            else:
                print(f"      [!] Weights not found for {pipeline_name}. Skipping evaluation.")
                continue
                
            screwdriver.eval()

            # Run the model against the ZERO-SHOT benchmarks
            for eval_name, config_data in EVAL_CONFIGS.items():
                task_config = {"task_name": eval_name, **config_data}
                
                metrics = evaluate_model(
                    task_config, screwdriver, large_model, small_model, 
                    harvester, tokenizer, device, iteration, pipeline_name, 
                    eval_samples=eval_samples
                )
                
                print(f"      => {eval_name} Success Rate: {metrics['success_rate_percent']:.1f}%")
                print(f"      => Mean Convergence Shift:  {metrics['mean_convergence_shift']:+.4f}")
            
            # Clean VRAM for the next pipeline
            del small_model, large_model, harvester, screwdriver
            torch.cuda.empty_cache()

    print("\n[+] 20-LOOP BENCHMARK COMPLETE.")

if __name__ == "__main__":
    main()
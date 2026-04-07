import os
import json
import torch
from datetime import datetime
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, log_loss
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as BuildDatasetMain
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.Tools import inject_weights, remove_weights
from ScrewDriver.ScrewDriverTrain import main as ScrewdriverTrainMain

def log_evaluation(eval_task_config, metrics, iteration, log_dir="eval_logs"):
    """Saves the highly detailed JSON telemetry for a specific evaluation run."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"loop{iteration}_eval_{eval_task_config['task_name']}_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "loop_iteration": iteration,
        "configuration": eval_task_config,
        "metrics": metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4)

def evaluate_model(eval_task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, iteration, eval_samples=500):
    """Runs the physical evaluation for exactly 500 samples."""
    print(f"      Running {eval_samples}-Sample Evaluation on {eval_task_config['task_name']}...")
    
    # 1. LOAD DATA
    if eval_task_config['config_name']:
        dataset = load_dataset(eval_task_config['dataset_path'], eval_task_config['config_name'], split="validation")
    else:
        dataset = load_dataset(eval_task_config['dataset_path'], split="test")

    if eval_task_config['task_name'] == "imdb_sentiment":
        short_texts = [row for row in dataset if len(row['text'].split()) < 100]
        cat_0 = [row for row in short_texts if row['label'] == 0][:eval_samples//2]
        cat_1 = [row for row in short_texts if row['label'] == 1][:eval_samples//2]
        combined = cat_0 + cat_1
    else:
        cat_0 = [row for row in dataset if row['label'] == 0][:eval_samples//2]
        cat_1 = [row for row in dataset if row['label'] == 1][:eval_samples//2]
        combined = cat_0 + cat_1
        
    texts = [row['sentence'] if 'sentence' in row else row['text'] for row in combined]
    labels = [row['label'] for row in combined]

    prompt_emb = harvester.embed_prompt(eval_task_config['task_label']).unsqueeze(0).to(device)
    base_features, steered_features = [], []

    # 2. INFERENCE LOOP
    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            base_features.append(large_model(**inputs).pooler_output.cpu().numpy())

        A_small_b, B_small_b, _ = harvester.extract_task_matrices(
            small_model, [eval_task_config['baseline_prompt'] + " " + text], 
            [eval_task_config['task_label'] + " " + text], is_small=True, calc_variance=False
        )
        
        with torch.no_grad():
            A_large, B_large, gate = screwdriver(A_small_b[0].unsqueeze(0).to(device), B_small_b[0].unsqueeze(0).to(device), prompt_emb, hard=True)
            active_layers = torch.where(gate.squeeze() > 0.5)[0].tolist()
            delta_W_seq = torch.matmul(B_large.squeeze(0), A_large.squeeze(0)).transpose(-1, -2)

        injected_weights = []
        prev_layer = -999
        for l_idx in active_layers:
            dist = l_idx - prev_layer
            decay = 1.0 if dist > 100 else 1.0 - (0.3 * np.exp(-0.3 * (dist - 1)))
            scaled_W = 3.0 * decay * delta_W_seq[l_idx]
            inject_weights(large_model, l_idx, scaled_W)
            injected_weights.append((l_idx, scaled_W))
            prev_layer = l_idx

        with torch.no_grad():
            steered_features.append(large_model(**inputs).pooler_output.cpu().numpy())

        for l_idx, w in injected_weights:
            remove_weights(large_model, l_idx, w)

    # 3. LINEAR PROBE
    X_base = np.vstack(base_features)
    X_steered = np.vstack(steered_features)

    # Clean out any stray NaNs (replace with 0)
    if np.isnan(X_base).any() or np.isnan(X_steered).any():
        print("      [!] Warning: NaNs detected in features. Cleaning...")
        X_base = np.nan_to_num(X_base)
        X_steered = np.nan_to_num(X_steered)

    X_b_train, X_b_test, y_train, y_test = train_test_split(X_base, labels, test_size=0.2, random_state=42)
    X_s_train, X_s_test, _, _ = train_test_split(np.vstack(steered_features), labels, test_size=0.2, random_state=42)

    clf_base = LogisticRegression(max_iter=1000).fit(X_b_train, y_train)
    clf_steered = LogisticRegression(max_iter=1000).fit(X_s_train, y_train)

    base_acc = accuracy_score(y_test, clf_base.predict(X_b_test))
    steered_acc = accuracy_score(y_test, clf_steered.predict(X_s_test))

    base_loss = log_loss(y_test, clf_base.predict_proba(X_b_test))
    steered_loss = log_loss(y_test, clf_steered.predict_proba(X_s_test))

    metrics = {
        "base_accuracy": base_acc,
        "steered_accuracy": steered_acc,
        "accuracy_shift": steered_acc - base_acc,
        "base_log_loss": base_loss,
        "steered_log_loss": steered_loss,
        "log_loss_shift": steered_loss - base_loss,
        "average_layers_altered": len(active_layers) # Quick snapshot from the final sentence
    }

    # Save the detailed JSON log
    log_evaluation(eval_task_config, metrics, iteration)

    return {
        "base_acc": base_acc,
        "steered_acc": steered_acc,
        "shift": steered_acc - base_acc
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define the core setups
    CONFIGS = {
        "imdb_sentiment": {
            "dataset_path": "imdb", "config_name": None,
            "task_label": "Analyze the sentiment of this text:",
            "baseline_prompt": "The event occurred on a Tuesday afternoon."
        },
        "ag_news_classification": {
            "dataset_path": "ag_news", "config_name": None,
            "task_label": "Classify the news topic of this article:",
            "baseline_prompt": "The event occurred on a Tuesday afternoon."
        },
        "glue_sst2": {
            "dataset_path": "glue", "config_name": "sst2",
            "task_label": "Determine if the emotion is positive or negative:",
            "baseline_prompt": "The event occurred on a Tuesday afternoon."
        }
    }

    # The 4 Pipelines to run per loop
    PIPELINES = ["imdb_sentiment","ag_news_classification","glue_sst2","combined"]
    
    learning_rate = 5e-5
    sparsity_lambda = 0.05
    eval_samples = 500

    # ==========================================
    # THE 20-LOOP MASTER GAUNTLET
    # ==========================================
    for iteration in range(1, 21):
        print("\n" + "#"*60)
        print(f"      STARTING MASTER LOOP {iteration} OF 20")
        print("#"*60)
        
        for pipeline_name in PIPELINES:
            print(f"\n{'='*40}\nEXECUTING FULL PIPELINE: {pipeline_name.upper()}\n{'='*40}")
            
            # --- 1. BUILD DATASET ---
            print(f"  [1/3] Building Dataset for {pipeline_name}...")
            BuildDatasetMain()
            
            # --- 2. TRAIN SCREWDRIVER ---
            print(f"\n  [2/3] Training Screwdriver for {pipeline_name}...")
            ScrewdriverTrainMain(task_name=pipeline_name, task_label=pipeline_name)
            
            # --- 3. EVALUATE ---
            print(f"\n  [3/3] Evaluating Screwdriver for {pipeline_name}...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
            large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
            harvester = Harvester(small_model, large_model, tokenizer, device=device)
            
            screwdriver = ModelScrewDriver(d_small=768, d_large=1024, d_prompt=768, target_rank=8, num_small_layers=12, num_large_layers=24).to(device)
            weights_path = f"ModelScrewdriver_{pipeline_name}.pth"
            screwdriver.load_state_dict(torch.load(weights_path, weights_only=True))
            screwdriver.eval()

            loop_results = {}
            
            # Specific task evaluation
            if pipeline_name in CONFIGS:
                task_config = {"task_name": pipeline_name, **CONFIGS[pipeline_name]}
                metrics = evaluate_model(task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, iteration, eval_samples=eval_samples)
                loop_results[pipeline_name] = metrics
                print(f"      Shift: {metrics['shift']*100:+.2f}%")
                
            # Generalist combined evaluation
            elif pipeline_name == "combined":
                for eval_name, config_data in CONFIGS.items():
                    task_config = {"task_name": eval_name, **config_data}
                    metrics = evaluate_model(task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, iteration, eval_samples=eval_samples)
                    loop_results[eval_name] = metrics
                    print(f"      {eval_name} Shift: {metrics['shift']*100:+.2f}%")
            
            # Nuke VRAM
            del small_model, large_model, harvester, screwdriver
            torch.cuda.empty_cache()

    print("\n[+] 20-LOOP BENCHMARK COMPLETE. Check 'benchmark_logs/master_benchmark_log.txt'.")

if __name__ == "__main__":
    main()
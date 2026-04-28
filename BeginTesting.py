import os
import json
import torch
import time
import random
from datetime import datetime
import numpy as np
from datasets import load_dataset
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer

from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as BuildDatasetMain
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.Tools import inject_weights, remove_weights
from ScrewDriver.ScrewDriverTrain import start as ScrewdriverTrainMain

def log_evaluation(metrics, iteration, log_dir="eval_logs"):
    """Saves the highly detailed JSON telemetry for a specific evaluation run."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"loop{iteration}_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "loop_iteration": iteration,
        "metrics": metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4)

def evaluate_model(eval_task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, iteration, model_name, eval_samples=500):
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
    
    random.shuffle(valid_data)
    
    valid_data = valid_data[:eval_samples]
    
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
            k_val = 3
            active_layers = torch.topk(gate.squeeze(), k=k_val).indices.tolist()
            active_layers_snapshot = active_layers
            delta_W_seq = torch.matmul(B_large.squeeze(0), A_large.squeeze(0)).transpose(-1, -2)

        # C. Inject Weights
        injected_weights = []
        for l_idx in active_layers:
            scaled_W = delta_W_seq[l_idx]
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
    unique_labels = len(set(labels))
    unique_labels = len(set(labels))
    if unique_labels < 2:
        print(f"      [!] Warning: Only found {unique_labels} unique label(s). Bypassing metrics.")
        metrics = {
            "base_cluster_score": 0.0, "steered_cluster_score": 0.0, "cluster_improvement": 0.0,
            "base_ari": 0.0, "steered_ari": 0.0, "ari_improvement": 0.0,
            "base_accuracy": 0.0, "steered_accuracy": 0.0, "accuracy_improvement": 0.0,
            "base_f1": 0.0, "steered_f1": 0.0, "f1_improvement": 0.0,
            "average_layers_altered": len(active_layers_snapshot)
        }
        return metrics
    
    # Calculate how tightly the data naturally clusters based on the labels
    base_silhouette = silhouette_score(X_base, labels)
    steered_silhouette = silhouette_score(X_steered, labels)
    
    kmeans_base = KMeans(n_clusters=unique_labels, random_state=1068, n_init='auto').fit(X_base)
    kmeans_steered = KMeans(n_clusters=unique_labels, random_state=1068, n_init='auto').fit(X_steered)
    
    ari_base = adjusted_rand_score(labels, kmeans_base.labels_)
    ari_steered = adjusted_rand_score(labels, kmeans_steered.labels_)

    # C. Linear Probing (Downstream Accuracy & F1)
    # 80/20 train/test split of our zero-shot sample to simulate downstream fine-tuning
    X_train_b, X_test_b, y_train, y_test = train_test_split(X_base, labels, test_size=0.2, random_state=1068, stratify=labels)
    X_train_s, X_test_s, _, _ = train_test_split(X_steered, labels, test_size=0.2, random_state=1068, stratify=labels)
    
    clf_base = LogisticRegression(max_iter=1000).fit(X_train_b, y_train)
    clf_steered = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    
    y_pred_b = clf_base.predict(X_test_b)
    y_pred_s = clf_steered.predict(X_test_s)
    
    base_acc = accuracy_score(y_test, y_pred_b)
    steered_acc = accuracy_score(y_test, y_pred_s)
    
    # Macro F1 handles potential class imbalances gracefully
    base_f1 = f1_score(y_test, y_pred_b, average='macro')
    steered_f1 = f1_score(y_test, y_pred_s, average='macro')

    metrics = {
        "base_cluster_score": float(base_silhouette),
        "steered_cluster_score": float(steered_silhouette),
        "cluster_improvement": float(steered_silhouette - base_silhouette),
        
        "base_ari": float(ari_base),
        "steered_ari": float(ari_steered),
        "ari_improvement": float(ari_steered - ari_base),
        
        "base_accuracy": float(base_acc),
        "steered_accuracy": float(steered_acc),
        "accuracy_improvement": float(steered_acc - base_acc),
        
        "base_f1": float(base_f1),
        "steered_f1": float(steered_f1),
        "f1_improvement": float(steered_f1 - base_f1),
        
        "average_layers_altered": len(active_layers_snapshot)
    }

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
        },
        # --- NEW CONFIGS BELOW ---
        "movie_sentiment": {
            "dataset_path": "rotten_tomatoes", "config_name": "default", "split": "validation",
            "task_label": "Analyze the sentiment of this text:",
            "baseline_prompt": "The window was left slightly open."
        },
        # "news_topic": {
        #     "dataset_path": "SetFit/bbc-news", "config_name": "default", "split": "test",
        #     "task_label": "Classify the news topic of this article:",
        #     "baseline_prompt": "The bridge spans the entire river."
        # },
        "subjectivity": {
            "dataset_path": "SetFit/subj", "config_name": "default", "split": "test",
            "task_label": "Determine if this text is objective or subjective:",
            "baseline_prompt": "The receipt was printed on thermal paper."
        },
        "spam_detection": {
            "dataset_path": "SetFit/enron_spam", "config_name": "default", "split": "test",
            "task_label": "Categorize this text as legitimate or spam:",
            "baseline_prompt": "A stack of magazines sat on the coffee table."
        }
    }

    # Your existing training pipelines (We do not touch the master data)
    model_name = "011"
    
    eval_samples = 500
    
    # --- 1. BUILD DATASET (Uses your existing extraction logic) ---
    print(f"  [1/3] Building Dataset for {model_name}...")
    data_start = time.perf_counter()
    BuildDatasetMain()
    data_end = time.perf_counter()
    

    # ==========================================
    # THE MASTER GAUNTLET
    # ==========================================
    for iteration in range(1, 12):
        print("\n" + "#"*60)
        print(f"      STARTING MASTER LOOP {iteration} OF 20")
        print("#"*60)
        
        iteration_metrics = {}
        
        iteration_metrics['iteration'] = iteration
        iteration_metrics['eval_samples'] = eval_samples
        iteration_metrics['dataset build time'] = data_end-data_start
        
        print(f"\n{'='*40}\nEXECUTING FULL PIPELINE: {model_name.upper()}\n{'='*40}")
        pipe_start = time.perf_counter()
        
        # --- 2. TRAIN SCREWDRIVER ---
        print(f"\n  [2/3] Training Screwdriver for {model_name}...")
        train_start = time.perf_counter()
        iteration_metrics['avg_w'], iteration_metrics['avg_r'] = ScrewdriverTrainMain(model_name=model_name, target_rank=iteration)
        train_end = time.perf_counter()
        
        iteration_metrics['Screwdriver training time'] = train_end-train_start
        
        # --- 3. ZERO-SHOT EVALUATION ---
        print(f"\n  [3/3] Evaluating Screwdriver for {model_name} on Unseen Cousins...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
        large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
        harvester = Harvester(small_model, large_model, tokenizer, device=device)
        
        screwdriver = ModelScrewDriver(d_small=768, d_large=1024, d_prompt=768, target_rank=iteration, num_large_layers=24).to(device)
        weights_path = f"ModelScrewdriver_{model_name}.pth"
        
        # Catch file not found if training failed
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, weights_only=True)
            screwdriver.load_state_dict(state_dict)
            
            iteration_metrics['internal_params'] = {
                "magnitude_scalar": float(state_dict['magnitude_scalar'].cpu()),
                "beta": float(state_dict['beta'].cpu()),
                "restore_mag": float(state_dict['restore_mag'].cpu())
            }
            
        else:
            print(f"      [!] Weights not found for {model_name}. Skipping evaluation.")
            continue
            
        
        screwdriver.eval()
        
        # Run the model against the ZERO-SHOT benchmarks
        for eval_name, config_data in EVAL_CONFIGS.items():
            
            task_config = {"task_name": eval_name, **config_data}
            
            eval_start = time.perf_counter()
            metrics = evaluate_model(
                task_config, screwdriver, large_model, small_model, 
                harvester, tokenizer, device, iteration, model_name, 
                eval_samples=eval_samples
            )
            eval_end = time.perf_counter()
            
            metrics['time to complete'] = eval_end-eval_start
            metrics['eval_config'] = config_data
            
            iteration_metrics[f'{eval_name} end stats'] = metrics
            
            print(f"      => {eval_name} Base Silhouette:    {metrics['base_cluster_score']:.4f}")
            print(f"      => {eval_name} Steered Silhouette: {metrics['steered_cluster_score']:.4f}")
            print(f"      => {eval_name} Cluster Imprv:      {metrics['cluster_improvement']:+.4f}")
            print(f"      --------------------------------------------------")
            print(f"      => {eval_name} Base Accuracy:      {metrics['base_accuracy']:.2%}")
            print(f"      => {eval_name} Steered Accuracy:   {metrics['steered_accuracy']:.2%}")
            print(f"      => {eval_name} Acc Imprv:          {metrics['accuracy_improvement']:+.2%}\n")
        
        # Clean VRAM for the next pipeline
        del small_model, large_model, harvester, screwdriver
        torch.cuda.empty_cache()
        
        pipe_end = time.perf_counter()
        
        iteration_metrics['total time'] = pipe_end-pipe_start
        
        log_evaluation(iteration_metrics, iteration)

    print("\n[+] 20-LOOP BENCHMARK COMPLETE.")

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertTokenizer

from ScrewDriver.ScrewDriver import ModelScrewDriver
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.Tools import inject_weights, remove_weights

def extract_features(texts, model, tokenizer, screwdriver, harvester, device, task_config, use_screwdriver=False):
    """Extracts embeddings. If use_screwdriver=True, it dynamically steers BERT per sentence."""
    features = []
    
    # Pre-embed the prompt so the Screwdriver knows the task
    prompt_emb = harvester.embed_prompt(task_config['task_label']).unsqueeze(0).to(device)

    for idx, text in enumerate(texts):
        if idx % 100 == 0 and idx > 0:
            print(f"      Processed {idx}/{len(texts)} samples...")
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        injected_weights = []
        
        # --- SCREWDRIVER INJECTION (Soft Routing) ---
        if use_screwdriver:
            A_small_b, B_small_b, _ = harvester.extract_task_matrices(
                harvester.small_model, [task_config['baseline_prompt'] + " " + text], 
                [task_config['task_label'] + " " + text], is_small=True, calc_variance=False
            )
            
            with torch.inference_mode():
                A_large, B_large, gate = screwdriver(
                    A_small_b[0].unsqueeze(0).to(device), 
                    B_small_b[0].unsqueeze(0).to(device), 
                    prompt_emb, 
                    hard=False 
                )
                
                # Using the exact same Top-K=3 that proved successful in clustering
                active_layers = torch.topk(gate.squeeze(), k=3).indices.tolist()
                delta_W_seq = torch.matmul(B_large.squeeze(0), A_large.squeeze(0)).mT

            for l_idx in active_layers:
                # Inject with the 0.5 dampener
                scaled_W = delta_W_seq[l_idx] * 0.5
                inject_weights(model, l_idx, scaled_W)
                injected_weights.append((l_idx, scaled_W))

        # --- INFERENCE ---
        with torch.inference_mode():
            features.append(model(**inputs).pooler_output.cpu().numpy())

        # --- CLEANUP ---
        if use_screwdriver:
            for l_idx, w in injected_weights:
                remove_weights(model, l_idx, w)

    return np.vstack(features)


def run_accuracy_benchmark(task_config, screwdriver, large_model, tokenizer, harvester, device, train_samples=800, test_samples=200):
    print(f"\n{'='*50}\nSTARTING DOWNSTREAM ACCURACY BENCHMARK: {task_config['task_name']}\n{'='*50}")
    
    # 1. LOAD DATASET
    print("[*] Loading Dataset...")
    if task_config.get('config_name'):
        dataset = load_dataset(task_config['dataset_path'], task_config['config_name'], split=task_config['split'])
    else:
        dataset = load_dataset(task_config['dataset_path'], split=task_config['split'])

    # Extract text and labels, ensuring they exist
    valid_data = [(row.get('sentence', row.get('text')), row['label']) for row in dataset if row.get('label') is not None]
    
    # Filter for length
    valid_data = [(t, l) for t, l in valid_data if len(t.split()) < 100]
    
    # --- ADD THESE TWO LINES ---
    import random
    random.shuffle(valid_data)
    # ---------------------------
    
    # Now it will grab a mixed bag of 800 samples!
    train_data = valid_data[:train_samples]
    test_data = valid_data[train_samples : train_samples + test_samples]
    
    X_train_texts, y_train = [item[0] for item in train_data], [item[1] for item in train_data]
    X_test_texts, y_test = [item[0] for item in test_data], [item[1] for item in test_data]

    # 2. EXTRACT BASE FEATURES (The Control)
    print("\n[+] Extracting BASELINE Features...")
    X_train_base = extract_features(X_train_texts, large_model, tokenizer, screwdriver, harvester, device, task_config, use_screwdriver=False)
    X_test_base = extract_features(X_test_texts, large_model, tokenizer, screwdriver, harvester, device, task_config, use_screwdriver=False)

    # 3. EXTRACT STEERED FEATURES (The Screwdriver)
    print("\n[+] Extracting STEERED Features (Applying Weights)...")
    X_train_steered = extract_features(X_train_texts, large_model, tokenizer, screwdriver, harvester, device, task_config, use_screwdriver=True)
    X_test_steered = extract_features(X_test_texts, large_model, tokenizer, screwdriver, harvester, device, task_config, use_screwdriver=True)

    # 4. TRAIN CLASSIFIERS & EVALUATE
    print("\n[*] Training Linear Probes...")
    
    clf_base = LogisticRegression(max_iter=1000)
    clf_base.fit(X_train_base, y_train)
    base_preds = clf_base.predict(X_test_base)
    base_acc = accuracy_score(y_test, base_preds)

    clf_steered = LogisticRegression(max_iter=1000)
    clf_steered.fit(X_train_steered, y_train)
    steered_preds = clf_steered.predict(X_test_steered)
    steered_acc = accuracy_score(y_test, steered_preds)

    # 5. THE VERDICT
    print(f"\n{'='*50}\nFINAL BENCHMARK RESULTS: {task_config['task_name']}\n{'='*50}")
    print(f"Base BERT Accuracy:     {base_acc * 100:.2f}%")
    print(f"Steered BERT Accuracy:  {steered_acc * 100:.2f}%")
    
    accuracy_shift = (steered_acc - base_acc) * 100
    print(f"Net Accuracy Shift:     {accuracy_shift:+.2f}%")
    
    if accuracy_shift > 0:
        print("\n[SUCCESS] The Screwdriver successfully improved downstream task execution!")
    else:
        print("\n[NOTE] The Screwdriver did not improve absolute accuracy on this sample.")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Downstream Evaluator on {device}...")

    # Load Base Models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    harvester = Harvester(small_model, large_model, tokenizer, device=device)
    
    # Load the trained Screwdriver (Ensure target_rank matches your trained weights, 8 or 16)
    screwdriver = ModelScrewDriver(d_small=768, d_large=1024, d_prompt=768, target_rank=8, num_small_layers=12, num_large_layers=24).to(device)
    
    # We will load the IMDB weights to test In-Domain first
    weights_path = "ModelScrewdriver_imdb_sentiment.pth" 
    if os.path.exists(weights_path):
        screwdriver.load_state_dict(torch.load(weights_path, weights_only=True))
        screwdriver.eval()
        print(f"[+] Successfully loaded weights from {weights_path}")
    else:
        print(f"[!] Critical Error: {weights_path} not found. Train the model first.")
        return

    # ==========================================
    # EVALUATION CONFIGURATIONS
    # ==========================================
    # 1. IN-DOMAIN SANITY CHECK (Trained on this)
    imdb_config = {
        "task_name": "imdb_sentiment",
        "dataset_path": "imdb",
        "config_name": None,
        "split": "test", # Use the test split since it trained on the train split
        "task_label": "Analyze the sentiment of this text:",
        "baseline_prompt": "The event occurred on a Tuesday afternoon."
    }

    # 2. ZERO-SHOT COUSIN CHECK (Never seen this)
    finance_config = {
        "task_name": "finance_sentiment",
        "dataset_path": "FinanceMTEB/financial_phrasebank",
        "config_name": "default",
        "split": "train",
        "task_label": "Analyze the sentiment of this text:",
        "baseline_prompt": "The event occurred on a Tuesday afternoon."
    }

    # Run the In-Domain test first!
   # run_accuracy_benchmark(imdb_config, screwdriver, large_model, tokenizer, harvester, device)
    
    # Uncomment this when you are ready to test the Zero-Shot capabilities!
    run_accuracy_benchmark(finance_config, screwdriver, large_model, tokenizer, harvester, device)


if __name__ == "__main__":
    main()
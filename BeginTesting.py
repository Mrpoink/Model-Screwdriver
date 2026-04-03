import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from datasets import load_dataset

# Import your existing custom architecture
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.ScrewDriver import ModelScrewDriver
from ScrewDriverTrain import main as Screwdrivertrain

# --- SURGICAL INJECTION HELPERS ---
def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.add_(delta_W)

def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.sub_(delta_W)

# --- LOGGING ENGINE ---
def log_evaluation(config, metrics, log_dir="eval_logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"eval_{config['task_name']}_{timestamp}.json")
    
    log_data = {
        "timestamp": timestamp,
        "configuration": config,
        "metrics": metrics
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\n[+] Full evaluation securely logged to: {filepath}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. EVALUATION CONFIGURATION (This is what gets logged)
    config = {
        "task_name": "imdb_sentiment",
        "task_label": "Analyze the sentiment of this text.",
        "baseline_prompt": "The movie had a second act.",
        "screwdriver_weights": "ModelScrewdriver.pth",
        "eval_samples": 1000,
        "target_rank": 8,
        "beta": 0.3,
        "healing_rate": 0.3,
        "alpha_injection": 3.0, # Tune this based on your Rank-8 stability
        "learning_rate" : 5e-5
    }

    print("Loading Base Models and Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    
    harvester = Harvester(small_model, large_model, tokenizer, device=device)
    
    print(f"Loading Trained Screwdriver Engine ({config['screwdriver_weights']})...")
    screwdriver = ModelScrewDriver(
        d_small=768, d_large=1024, d_prompt=768, 
        target_rank=config['target_rank'], num_small_layers=12, num_large_layers=24
    ).to(device)
    
    
    Screwdrivertrain(config["learning_rate"])
    screwdriver.load_state_dict(torch.load(config['screwdriver_weights'], weights_only=True))
    screwdriver.eval()

   # 2. LOAD UNSEEN EVALUATION DATA
    print(f"Loading {config['eval_samples']} evaluation samples from {config['task_name']}...")
    dataset = load_dataset("imdb", split="test")
    
    # Filter for short reviews first
    short_reviews = [row for row in dataset if len(row['text'].split()) < 100]
    
    # Force a 50/50 split of positive and negative labels
    half_samples = config['eval_samples'] // 2
    pos_reviews = [row for row in short_reviews if row['label'] == 1][:half_samples]
    neg_reviews = [row for row in short_reviews if row['label'] == 0][:half_samples]
    
    combined_reviews = pos_reviews + neg_reviews
    
    # The train_test_split later on will automatically shuffle these for the classifier
    texts = [row['text'] for row in combined_reviews]
    labels = [row['label'] for row in combined_reviews]
    
    prompt_emb = harvester.embed_prompt(config['task_label']).unsqueeze(0).to(device)
    base_features, steered_features = [], []

    # 3. DYNAMIC INFERENCE LOOP
    print("\n--- Running Dynamic Steering Evaluation ---")
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"  [*] Processing {idx}/{len(texts)}...")
            
        # BASE INFERENCE
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            base_features.append(large_model(**inputs).pooler_output.cpu().numpy())

        # STEERED INFERENCE
        A_small_b, B_small_b, _ = harvester.extract_task_matrices(
            small_model, [config['baseline_prompt']], [text], is_small=True
        )
        
        with torch.no_grad():
            A_large, B_large, gate = screwdriver(
                A_small_b[0].unsqueeze(0).to(device), 
                B_small_b[0].unsqueeze(0).to(device), 
                prompt_emb, hard=True
            )
            active_layers = torch.where(gate.squeeze() > 0.5)[0].tolist()
            delta_W_seq = torch.matmul(B_large.squeeze(0), A_large.squeeze(0)).transpose(-1, -2)

        injected_weights = []
        prev_layer = -999
        for l_idx in active_layers:
            dist = l_idx - prev_layer
            decay = 1.0 if dist > 100 else 1.0 - (config["beta"] * np.exp(-config["healing_rate"] * (dist - 1)))
            scaled_W = config["alpha_injection"] * decay * delta_W_seq[l_idx]
            
            inject_weights(large_model, l_idx, scaled_W)
            injected_weights.append((l_idx, scaled_W))
            prev_layer = l_idx

        with torch.no_grad():
            steered_features.append(large_model(**inputs).pooler_output.cpu().numpy())

        # CLEANUP
        for l_idx, w in injected_weights:
            remove_weights(large_model, l_idx, w)

    # 4. LINEAR PROBE & METRICS
    print("\n--- Training Linear Probes ---")
    X_base_train, X_base_test, y_train, y_test = train_test_split(np.vstack(base_features), labels, test_size=0.2, random_state=42)
    X_steered_train, X_steered_test, _, _ = train_test_split(np.vstack(steered_features), labels, test_size=0.2, random_state=42)

    clf_base = LogisticRegression(max_iter=1000).fit(X_base_train, y_train)
    clf_steered = LogisticRegression(max_iter=1000).fit(X_steered_train, y_train)

    base_acc = accuracy_score(y_test, clf_base.predict(X_base_test))
    steered_acc = accuracy_score(y_test, clf_steered.predict(X_steered_test))
    base_loss = log_loss(y_test, clf_base.predict_proba(X_base_test))
    steered_loss = log_loss(y_test, clf_steered.predict_proba(X_steered_test))

    # 5. CONSOLE REPORT
    print("\n" + "="*40)
    print("      END-TO-END EVALUATION REPORT")
    print("="*40)
    print(f"Base Accuracy:    {base_acc * 100:.2f}%")
    print(f"Steered Accuracy: {steered_acc * 100:.2f}%")
    print(f"Accuracy Shift:   {(steered_acc - base_acc) * 100:+.2f}%")
    print("-" * 40)
    print(f"Base Log Loss:    {base_loss:.4f}")
    print(f"Steered Log Loss: {steered_loss:.4f}")
    print("="*40)

    # 6. SAVE TO LOG
    metrics = {
        "base_accuracy": base_acc,
        "steered_accuracy": steered_acc,
        "accuracy_shift": steered_acc - base_acc,
        "base_log_loss": base_loss,
        "steered_log_loss": steered_loss,
        "log_loss_shift": steered_loss - base_loss,
        "average_layers_altered": len(active_layers) # Quick snapshot of routing sparsity
    }
    
    log_evaluation(config, metrics)

if __name__ == "__main__":
    for i in range(20):
        main()
        torch.cuda.empty_cache()
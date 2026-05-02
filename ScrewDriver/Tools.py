import torch
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import random
from datasets import load_dataset
class AdaptiveSpikeScheduler:
    def __init__(self, optimizer, base_lr, min_lr=1e-6, spike_tolerance=1.5, plateau_patience=5, drop_factor=0.5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr  # NEW: The hard floor to prevent paralysis
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.spike_tolerance = spike_tolerance 
        self.plateau_patience = plateau_patience
        self.drop_factor = drop_factor

    def step(self, current_loss):
        if current_loss > self.best_loss * self.spike_tolerance:
            self._multiply_lr(self.drop_factor)
            print(f"      [!] Manifold Spike! Slashed LR to {self._get_lr():.2e}")
            self.best_loss = current_loss 
            self.patience_counter = 0
            return

        if current_loss < self.best_loss * 0.98:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.plateau_patience:
            self._multiply_lr(self.drop_factor)
            print(f"      [-] Loss Plateau. Fine-tuning LR to {self._get_lr():.2e}")
            self.patience_counter = 0
            
    def _multiply_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            # Drop the LR, but NEVER let it fall below min_lr
            new_lr = max(param_group['lr'] * factor, self.min_lr)
            param_group['lr'] = new_lr
            
    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reset_phase(self):
        """NEW: Wipes the memory and restores the base learning rate for a new phase."""
        self.best_loss = float('inf')
        self.patience_counter = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.add_(delta_W)

def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    model.encoder.layer[layer_idx].attention.output.dense.weight.data.sub_(delta_W)

def evaluate_model(eval_task_config, screwdriver, large_model, small_model, harvester, tokenizer, device, eval_samples=1000):
    """Function to evaluated model

    Args:
        eval_task_config (dict): Configuration for test, must include eval name key with values: dataset_path, task_label, baseline_prompt
        screwdriver (Screwdriver): Screwdriver model
        large_model (BertModel): target model for weight change
        small_model (BertModel): scout model for weight extraction
        harvester (Harvester): harvester object to view internal represenations of scout model
        tokenizer (BertTokenizer): text tokenizer
        device (torch.device): torch.device()
        eval_samples (int, optional): number of samples per evaluation. Defaults to 1000.

    Returns:
        Dict: metrics for all evaluations within configuration
    """
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

    # Linear Probing (Downstream Accuracy & F1)
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

def warm_up_teacher(model, train_loader, val_loader, target_acc=0.90, max_epochs=5):
    """
    Trains the large model (preferably via LoRA) until it reaches a 
    specific accuracy, providing a 'Gold Signal' for harvesting.
    """
    print(f"[*] Warming up Teacher Model to {target_acc*100}% accuracy...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(max_epochs):
        model.train()
        # standard training loop logic...
        
        # Validation Check
        val_acc = evaluate_accuracy(model, val_loader)
        print(f"      Epoch {epoch} | Val Acc: {val_acc:.2%}")
        
        if val_acc >= target_acc:
            print(f"[!] Target reached. Freezing Teacher for harvesting.")
            break
    return model
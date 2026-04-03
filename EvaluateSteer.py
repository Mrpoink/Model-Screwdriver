import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.ScrewDriver import ModelScrewDriver

# --- INJECTION HELPERS ---
def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    with torch.no_grad():
        target_module.weight.data.add_(delta_W)

def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    with torch.no_grad():
        target_module.weight.data.sub_(delta_W)

def get_imdb_subset(num_samples=1000):
    print("Loading IMDB Evaluation Dataset...")
    dataset = load_dataset("imdb", split="test")
    
    # Get short reviews for faster processing
    short_reviews = [row for row in dataset if len(row['text'].split()) < 100]
    
    pos_reviews = [row for row in short_reviews if row['label'] == 1][:num_samples // 2]
    neg_reviews = [row for row in short_reviews if row['label'] == 0][:num_samples // 2]
    
    combined = pos_reviews + neg_reviews
    np.random.shuffle(combined)
    
    texts = [row['text'] for row in combined]
    labels = [row['label'] for row in combined]
    return texts, labels

def extract_features_in_batches(model, tokenizer, texts, device, batch_size=32):
    features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            # Use pooler_output for classification
            features.append(outputs.pooler_output.cpu().numpy())
    return np.vstack(features)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Models...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    
    harvester = Harvester(small_model, large_model, tokenizer, device=device)
    
    print("Loading Screwdriver Engine...")
    screwdriver = ModelScrewDriver(
        d_small=768, d_large=1024, target_rank=8, d_prompt=768, 
        num_small_layers=12, num_large_layers=24
    ).to(device)
    screwdriver.load_state_dict(torch.load("ModelScrewdriver.pth", weights_only=True))
    screwdriver.eval()

    # 1. Get Test Data
    texts, labels = get_imdb_subset(num_samples=1000)
    
    # 2. Extract Base Features
    print("\n--- Extracting Base Model Features ---")
    base_features = extract_features_in_batches(large_model, tokenizer, texts, device)

    # 3. Forge and Inject Steered Weights
    print("\n--- Dynamically Extracting Steered Features ---")
    
    task_label = "Analyze the sentiment of this text."
    baseline_prompt = "The movie had a second act."
    prompt_emb = harvester.embed_prompt(task_label).unsqueeze(0).to(device)
    
    steered_features = []
    
    # We must crank the alpha to compensate for the Rank-1 bottleneck
    alpha = 1.0 
    beta, healing_rate = 0.5, 0.3
    
    # We must process sequentially so the Screwdriver can forge custom weights per prompt
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"  [*] Processing {idx}/{len(texts)}...")
            
        # 1. Scout the unique geometry of THIS prompt
        A_small_batch, B_small_batch, _ = harvester.extract_task_matrices(
            small_model, [baseline_prompt], [text], is_small=True
        )
        A_small = A_small_batch[0].unsqueeze(0).to(device)
        B_small = B_small_batch[0].unsqueeze(0).to(device)

        # 2. Forge the dynamic weights
        with torch.no_grad():
            A_large, B_large, gate = screwdriver(A_small, B_small, prompt_emb, hard=True)
            active_layers = torch.where(gate.squeeze() > 0.5)[0].tolist()
            
            A_large = A_large.squeeze(0)
            B_large = B_large.squeeze(0)
            delta_W_sequence = torch.matmul(B_large, A_large).transpose(-1, -2)

        # 3. Inject
        previous_layer = -999
        injected_weights = []
        
        for layer_idx in active_layers:
            distance = layer_idx - previous_layer
            decay = 1.0 if distance > 100 else 1.0 - (beta * np.exp(-healing_rate * (distance - 1)))
            
            scaled_delta_W = alpha * decay * delta_W_sequence[layer_idx]
            
            inject_weights(large_model, layer_idx, scaled_delta_W)
            injected_weights.append((layer_idx, scaled_delta_W))
            previous_layer = layer_idx

        # 4. Infer
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = large_model(**inputs)
            steered_features.append(outputs.pooler_output.cpu().numpy())

        # 5. Clean the surgical field for the next prompt
        for layer_idx, weight in injected_weights:
            remove_weights(large_model, layer_idx, weight)

    steered_features = np.vstack(steered_features)

    # 5. Cleanup
    for layer_idx, weight in injected_weights:
        remove_weights(large_model, layer_idx, weight)

    # --- TRAIN LINEAR PROBES ---
    print("\n--- Training Linear Probes ---")
    # Split data
    X_base_train, X_base_test, y_train, y_test = train_test_split(base_features, labels, test_size=0.2, random_state=42)
    X_steered_train, X_steered_test, _, _ = train_test_split(steered_features, labels, test_size=0.2, random_state=42)

    # Train and Evaluate Base
    clf_base = LogisticRegression(max_iter=1000)
    clf_base.fit(X_base_train, y_train)
    base_acc = accuracy_score(y_test, clf_base.predict(X_base_test))

    # Train and Evaluate Steered
    clf_steered = LogisticRegression(max_iter=1000)
    clf_steered.fit(X_steered_train, y_train)
    steered_acc = accuracy_score(y_test, clf_steered.predict(X_steered_test))

    # --- RESULTS ---
    print("\n" + "="*40)
    print("      END-TO-END ACCURACY REPORT")
    print("="*40)
    print(f"Base BERT-Large Accuracy:    {base_acc * 100:.2f}%")
    print(f"Steered BERT-Large Accuracy: {steered_acc * 100:.2f}%")
    print("="*40)
    
    shift = steered_acc - base_acc
    if shift > 0:
        print(f"[+] Screwdriver IMPROVED downstream accuracy by {shift * 100:.2f}%")
    else:
        print(f"[-] Screwdriver DECREASED downstream accuracy by {abs(shift) * 100:.2f}%")
        
    from sklearn.metrics import log_loss

    # --- MEASURE CONFIDENCE (LOG LOSS) ---
    # Log loss penalizes the model for being unsure. Lower is better.
    base_probs = clf_base.predict_proba(X_base_test)
    steered_probs = clf_steered.predict_proba(X_steered_test)

    base_loss = log_loss(y_test, base_probs)
    steered_loss = log_loss(y_test, steered_probs)

    print("\n" + "="*40)
    print("      CONFIDENCE & RESOLUTION REPORT")
    print("="*40)
    print(f"Base Model Log Loss:    {base_loss:.4f}")
    print(f"Steered Model Log Loss: {steered_loss:.4f}")
    print("="*40)
    
    if steered_loss < base_loss:
        print(f"[+] SUCCESS: The Screwdriver made the model MORE confident in its internal logic.")
    else:
        print(f"[-] FAILURE: The Screwdriver made the model LESS confident.")

if __name__ == "__main__":
    main()
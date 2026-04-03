import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver.ScrewDriver import ModelScrewDriver

def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    with torch.no_grad():
        target_module.weight.data.add_(delta_W)

def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    with torch.no_grad():
        target_module.weight.data.sub_(delta_W)

def evaluate_screwdriver():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Base Models and Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    
    harvester = Harvester(small_model, large_model, tokenizer, device=device)
    
    print("Loading Trained Screwdriver Engine...")
    screwdriver = ModelScrewDriver(
        d_small=768, d_large=1024, rank=1, d_prompt=768, 
        num_small_layers=12, num_large_layers=24
    ).to(device)
    
    screwdriver.load_state_dict(torch.load("ModelScrewdriver.pth", weights_only=True))
    screwdriver.eval()

    # --- EVALUATION DATA ---
    task_label = "Analyze the sentiment of this text."
    baseline_prompt = "The movie had a second act."
    
    # Prompts the model has NEVER seen during training
    test_prompts = [
        "The pacing of the second act was completely unbearable.",
        "I absolutely loved the character development in this film.",
        "It was an okay experience, nothing special.",
        "The cinematography was breathtaking, but the plot was dull.",
        "Worst two hours of my life, completely unwatchable."
    ]

    prompt_emb = harvester.embed_prompt(task_label).unsqueeze(0).to(device)
    
    # --- METRICS TRACKERS ---
    total_shift = 0.0
    total_layers_altered = 0
    
    # Physical Constants for Alpha Decay
    beta = 0.5
    healing_rate = 0.3

    print("\n--- Commencing Evaluation ---")
    
    print("Calibrating Concept Axis...")
    # Define pure anchors for the concept
    pos_anchor = "The pacing of the movie was wonderful and I loved it."
    neg_anchor = "The pacing of the movie was terrible and I hated it."
    
    with torch.no_grad():
        pos_inputs = tokenizer(pos_anchor, return_tensors="pt").to(device)
        neg_inputs = tokenizer(neg_anchor, return_tensors="pt").to(device)
        
        pos_thought = large_model(**pos_inputs).pooler_output
        neg_thought = large_model(**neg_inputs).pooler_output
        
        # Create the directional vector for "Positive Sentiment"
        concept_axis = (pos_thought - neg_thought).squeeze(0)
    
    for idx, user_prompt in enumerate(test_prompts):
        print(f"\n[Test {idx+1}/{len(test_prompts)}] Prompt: '{user_prompt}'")
        
        # 1. Unsteered Baseline
        inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_thought = large_model(**inputs).pooler_output
            
        # 2. Scout Matrix Extraction (The Radar)
        A_small_batch, B_small_batch, _ = harvester.extract_task_matrices(
            small_model, [baseline_prompt], [user_prompt], is_small=True
        )
        A_small = A_small_batch[0].unsqueeze(0).to(device)
        B_small = B_small_batch[0].unsqueeze(0).to(device)
        
        # 3. Screwdriver Generation (The Forge & Brain)
        with torch.no_grad():
            # hard=True forces the STE to output strict 1s and 0s
            A_large, B_large, gate = screwdriver(A_small, B_small, prompt_emb, hard=True)
            
            # Identify which layers the Router chose to open
            # gate shape: (Batch, 24). We squeeze to 1D.
            active_layers = torch.where(gate.squeeze() > 0.5)[0].tolist()
            
            A_large = A_large.squeeze(0) 
            B_large = B_large.squeeze(0) 
            
            # Generate the matrices: (24, 1024, 1024)
            delta_W_sequence = torch.matmul(B_large, A_large).transpose(-1, -2)

        total_layers_altered += len(active_layers)
        print(f"  [*] Router selected {len(active_layers)} layers: {active_layers}")

        # 4. Surgical Injection with Exponential Recovery Decay
        previous_layer = -999
        injected_weights = [] # Keep track so we can clean up exactly what we injected
        
        for layer_idx in active_layers:
            distance = layer_idx - previous_layer
            
            if distance > 100:
                alpha = 1.0 
            else:
                alpha = 1.0 - (beta * torch.exp(torch.tensor(-healing_rate * (distance - 1)))).item()
                
            scaled_delta_W = alpha * delta_W_sequence[layer_idx]
            
            inject_weights(large_model, layer_idx, scaled_delta_W)
            injected_weights.append((layer_idx, scaled_delta_W))
            previous_layer = layer_idx

        # 5. Steered Inference
        with torch.no_grad():
            steered_thought = large_model(**inputs).pooler_output

        # --- CALCULATE ACCURACY (DIRECTIONAL SHIFT) ---
        # 1. How aligned was the original thought with positive sentiment?
        base_alignment = F.cosine_similarity(base_thought.squeeze(0), concept_axis, dim=0).item()
        
        # 2. How aligned is the steered thought with positive sentiment?
        steered_alignment = F.cosine_similarity(steered_thought.squeeze(0), concept_axis, dim=0).item()
        
        # 3. The absolute improvement toward the target concept
        accuracy_shift = steered_alignment - base_alignment
        
        print(f"  [*] Baseline Alignment: {base_alignment:.4f}")
        print(f"  [*] Steered Alignment:  {steered_alignment:.4f}")
        
        if accuracy_shift > 0:
            print(f"  [+] SUCCESS: Moved {accuracy_shift:.4f} units TOWARD the target concept.")
        else:
            print(f"  [-] FAILURE: Moved {abs(accuracy_shift):.4f} units AWAY from the target concept.")
        
        # 6. Calculate Shift
        shift_magnitude = torch.norm(steered_thought - base_thought).item()
        total_shift += shift_magnitude
        print(f"  [*] Manifold Shift (L2 Norm): {shift_magnitude:.4f}")

        # 7. Cleanup
        for layer_idx, weight in injected_weights:
            remove_weights(large_model, layer_idx, weight)

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("      EVALUATION SUMMARY")
    print("="*40)
    print(f"Average Manifold Shift:  {total_shift / len(test_prompts):.4f}")
    print(f"Average Layers Altered:  {total_layers_altered / len(test_prompts):.1f} / 24")
    print("="*40)

if __name__ == "__main__":
    evaluate_screwdriver()
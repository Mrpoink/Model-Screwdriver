import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from DataExtraction.TaskVectorHarvester import Harvester
from ScrewDriver import ScrewDriver

def inject_weights(model, layer_idx: int, delta_W: torch.Tensor):
    """Adds the delta_W weight to the current o_proj attention head in the layer at layer_idx

    Args:
        model (_type_): model to apply changes
        layer_idx (int): layer index for change
        delta_W (torch.Tensor): change in the expected head to steer towards desired output
    """
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    
    with torch.no_grad():
        target_module.weight.data.add_(delta_W)
        
    print(f"Successfully injected weights into Layer {layer_idx}.")



def remove_weights(model, layer_idx: int, delta_W: torch.Tensor):
    """Removes delta_W weight change from current o_proj attention head at layer_idx to prevent oversteering

    Args:
        model (_type_): model to apply change to
        layer_idx (int): layer index for change
        delta_W (torch.Tensor): change in the expected head to steer towards desired output
    """
    target_module = model.encoder.layer[layer_idx].attention.output.dense
    
    with torch.no_grad():
        target_module.weight.data.sub_(delta_W)
        
    print(f"[*] Successfully REMOVED weights from Layer {layer_idx}.")


def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    print("Loading Base Models and Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    small_model.eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device)
    large_model.eval()
    
    
    harvester = Harvester(small_model, large_model, tokenizer, device=device)
    
    print("Loading Trained Screwdriver...")
    screwdriver = ScrewDriver.ModelScrewDriver(
        d_small=768, 
        d_large=1024, 
        rank=1, 
        d_prompt=768, 
        num_small_layers=12, 
        num_large_layers=24
    ).to(device)
    
    ## WARNING: THIS ASSUMES THAT YOU ARE USING THE FRESHLY MADE DATASET. weights_only IS A SECURITY CHECK MADE BY PYTORCH!!
    screwdriver.load_state_dict(torch.load("ModelScrewdriver.pth", weights_only=True))
    screwdriver.eval()

    
    # The task we trained the Screwdriver on
    task_label = "Analyze the sentiment of this text."
    
    # The live user prompt we want to steer
    user_prompt = "The pacing of the second act was completely unbearable."
    
    # We use a neutral baseline to calculate the live shift
    baseline_prompt = "The movie had a second act."
    
    # We know from our radar during dataset creation that small model layer 6 handles sentiment
    ## TODO: THIS MUST BE DYNAMIC!!
    small_scout_layer = 6 
    
    print(f"\n[User Input]: '{user_prompt}'")

    print("\n--- Running Unsteered Baseline ---")
    
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        
        base_outputs = large_model(**inputs)
        base_thought = base_outputs.pooler_output 
        ## TODO: HAVE THIS PRINT


    print("\n--- Engaging Screwdriver ---")
    
    
    # Step A: Scout the geometry using the small model
    print("1. Scouting prompt geometry with BERT-Base...")
    
    A_small_batch, B_small_batch, small_scout_layer = harvester.extract_task_matrices(
        small_model, [baseline_prompt], [user_prompt]
    )
    
    A_small = A_small_batch[0].unsqueeze(0).to(device)
    B_small = B_small_batch[0].unsqueeze(0).to(device)
    
    # Step B: Embed the task context
    prompt_emb = harvester.embed_prompt(task_label).unsqueeze(0).to(device)
    
   
    # Step C: Forge the weights and predict the layer
    print("2. Forging BERT-Large weights and calculating routing vector...")
    
    with torch.no_grad():
        A_large, B_large, container_logits = screwdriver(A_small, B_small, prompt_emb)
        
        predicted_container = torch.argmax(container_logits, dim=-1).item()
        stride_layers = [i for i in range(24) if i % 2 == predicted_container]
        
        A_large = A_large.squeeze(0) # Shape: (12, 1, 1024)
        B_large = B_large.squeeze(0) # Shape: (12, 1024, 1)
        
        # PyTorch linear weights are stored as (out_features, in_features), so we transpose (.T)
        delta_W = (B_large @ A_large).T
        
        # Alpha controls the "strength" of the steering. 1.0 is standard.
        # This can be user-set but.... testing
        alpha = 1.0 
        
        delta_W_sequence = torch.matmul(B_large, A_large).transpose(-1, -2)
        scaled_delta_W = alpha * delta_W_sequence

    print(f"[*] Screwdriver targeting Container: {'Evens' if predicted_container == 0 else 'Odds'}")

    # Step D: Physically alter the massive model
    for i, layer_idx in enumerate(stride_layers):
        inject_weights(large_model, layer_idx, scaled_delta_W[i])

    print("\n--- Running Steered Inference ---")
    with torch.no_grad():
        steered_outputs = large_model(**inputs)
        steered_thought = steered_outputs.pooler_output

    # We calculate the Euclidean distance between the base thought and the steered thought
    # to prove the model's internal representation was actually altered by our injection.
    # This is done for testing and verification purposes
    shift_magnitude = torch.norm(steered_thought - base_thought).item()
    
    print(f"\n[Verification] Structural shift detected in final representation: {shift_magnitude:.4f}")
    
    if shift_magnitude > 0.0:
        print("[*] SUCCESS: The Screwdriver successfully seized and altered the model's latent manifold.")
    else:
        print("[!] FAILURE: The model's output did not change.")

    # Always clean up your weights!
    print("\n--- Cleaning up ---")
    for i, layer_idx in enumerate(stride_layers):
        remove_weights(large_model, layer_idx, scaled_delta_W[i])

if __name__ == "__main__":
    main()
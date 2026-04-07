import os
import random
import torch
from transformers import BertModel, BertTokenizer

from DataExtraction import TaskVectorHarvester
from DatasetBuildData import build_master_task_pool

def main(num_total_samples=50000, shard_size=2000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[*] Initializing Meta-Learning Harvester on {device}...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    harvester = TaskVectorHarvester.Harvester(small_model, large_model, tokenizer, device=device)

    # 1. LOAD THE TASK POOL
    # This downloads the datasets and applies your 15/15/35/35 probability weights
    TASK_POOL = build_master_task_pool()
    tasks = list(TASK_POOL.keys())
    weights = [TASK_POOL[t]["weight"] for t in tasks]

    # ==========================================
    # 2. PRE-CALCULATE ORACLE TRACES & EMBEDDINGS
    # ==========================================
    # We do this once per task so the massive loop runs instantly
    print("\n[*] Pre-calculating Oracle Causal Traces & Embeddings for all tasks...")
    task_variances = {}
    task_prompt_embs = {}

    for task_name, config in TASK_POOL.items():
        print(f"      Mapping Concept Circuit for: {task_name}")
        
        # Use a small subset to map the variance (Causal Tracing)
        sample_texts = config["data"][:32]
        sample_prompts = [config["prompts"][0] + " " + text for text in sample_texts]
        
        variance = harvester.causal_trace_variance(large_model, sample_prompts)
        task_variances[task_name] = variance.cpu()
        
        # Pre-embed all possible prompts in the ensemble
        embs = [harvester.embed_prompt(p).cpu() for p in config["prompts"]]
        task_prompt_embs[task_name] = embs

    # ==========================================
    # 3. MASSIVE EXTRACTION LOOP
    # ==========================================
    os.makedirs("master_dataset", exist_ok=True)
    current_shard = []
    shard_index = 0

    print(f"\n[*] Commencing Massive Weighted Extraction ({num_total_samples} samples)...")

    for i in range(1, num_total_samples + 1):
        if i % 50 == 0:
            print(f"      Harvesting sample {i}/{num_total_samples}...")

        # A. Probabilistically select Task & Pop Text
        chosen_task = random.choices(tasks, weights=weights, k=1)[0]
        task_config = TASK_POOL[chosen_task]

        if len(task_config["data"]) == 0:
            continue # Skip if this specific task's dataset is exhausted

        random_idx = random.randint(0, len(task_config["data"]) - 1)
        text_sample = task_config["data"].pop(random_idx)

        # B. Randomize Prompts for Maximum Generalization
        base_prompt = random.choice(task_config["neutral"])
        task_prompts_ensemble = task_config["prompts"]

        # C. Extract Small Matrices (Ensembled over the synonymous prompts)
        A_small, B_small = harvester.extract_ensembled_matrices(
            small_model, base_prompt, task_prompts_ensemble, text_sample
        )

        # D. Extract Large Matrices (The Target)
        active_prompt = random.choice(task_prompts_ensemble)
        A_large_batch, B_large_batch, _ = harvester.extract_task_matrices(
            large_model, 
            [base_prompt + " " + text_sample], 
            [active_prompt + " " + text_sample], 
            is_small=False,
            calc_variance=False 
        )

        # E. Package the Record
        # Everything is forced to .cpu() immediately to protect your VRAM
        record = {
            "task_type": chosen_task, 
            "A_small": A_small[0].cpu(),
            "B_small": B_small[0].cpu(),
            "prompt_emb": random.choice(task_prompt_embs[chosen_task]).clone(),
            "A_large": A_large_batch[0].cpu(),
            "B_large": B_large_batch[0].cpu(),
            "target_variance": task_variances[chosen_task].clone()
        }

        current_shard.append(record)

        # F. Flush to disk to protect System RAM
        if len(current_shard) >= shard_size:
            filepath = f"master_dataset/shard_{shard_index:04d}.pt"
            torch.save(current_shard, filepath)
            print(f"      [+] Flushed Shard {shard_index:04d} to disk ({shard_size} samples).")
            current_shard = []
            shard_index += 1

    # Save any remaining stragglers
    if current_shard:
        filepath = f"master_dataset/shard_{shard_index:04d}.pt"
        torch.save(current_shard, filepath)
        print(f"      [+] Flushed final Shard {shard_index:04d} to disk.")

    print("\n[+] Massive Task-Space Extraction Complete!")

if __name__ == "__main__":
    main(num_total_samples=50000) # Adjust this number based on how long you want to let it run
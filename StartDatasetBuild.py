import os
import random
import torch
import shutil
import time
from transformers import BertModel, BertTokenizer

from DataExtraction.TaskVectorHarvester import Harvester # Fixed import to match your filename
from DataExtraction.ModelWarmup import warm_up_teacher
from DatasetBuildData import build_master_task_pool

def main(model_name, num_total_samples=15000, shard_size=1500):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Robust folder reset
    if os.path.exists(f"master_dataset{model_name}"):
        model_name += 1
    
    os.makedirs(f"master_dataset{model_name}", exist_ok=True)
    
    print(f"\n[*] Initializing Meta-Learning Harvester on {device}...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    harvester = Harvester(small_model, large_model, tokenizer, device=device)

    TASK_POOL = build_master_task_pool()
    tasks = list(TASK_POOL.keys())
    weights = [TASK_POOL[t]["weight"] for t in tasks]
    
    max_available_samples = sum(len(config["data"]) for config in TASK_POOL.values())
    run_samples = min(num_total_samples, max_available_samples)

    print("\n[*] Pre-calculating Oracle Causal Traces & Embeddings for all tasks...")
    task_variances = {}
    task_prompt_embs = {}

    # Use inference_mode and autocast to massively accelerate pre-calculation
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
        for task_name, config in TASK_POOL.items():
            print(f"      Mapping Concept Circuit for: {task_name}")
            
            sample_texts = config["data"][:32]
            sample_prompts = [config["prompts"][0] + " " + text for text, label in sample_texts]
            
            variance = harvester.causal_trace_variance(large_model, sample_prompts)
            task_variances[task_name] = variance.cpu()
            
            embs = [harvester.embed_prompt(p).cpu() for p in config["prompts"]]
            task_prompt_embs[task_name] = embs

    os.makedirs("master_dataset", exist_ok=True)
    current_shard = []
    shard_index = 0

    print(f"\n[*] Commencing Massive Weighted Extraction ({run_samples} samples)...")

    # Wrap the entire main loop in inference_mode and autocast for maximum throughput
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
        for i in range(1, run_samples + 1):
            if i % 50 == 0:
                print(f"      Harvesting sample {i}/{run_samples}...")

            chosen_task = random.choices(tasks, weights=weights, k=1)[0]
            task_config = TASK_POOL[chosen_task]

            if len(task_config["data"]) == 0:
                continue 

            random_idx = random.randint(0, len(task_config["data"]) - 1)
            text_sample, text_label = task_config["data"].pop(random_idx)

            base_prompt = random.choice(task_config["neutral"])
            task_prompts_ensemble = task_config["prompts"]
            
            sampled_ensemble = random.sample(task_prompts_ensemble, k=min(5, len(task_prompts_ensemble)))

            A_small, B_small = harvester.extract_ensembled_matrices(
                small_model, base_prompt, sampled_ensemble, text_sample
            )

            active_prompt = random.choice(task_prompts_ensemble)
            
            A_large_batch, B_large_batch, _ = harvester.extract_task_matrices(
                large_model, 
                [base_prompt + " " + text_sample], 
                [active_prompt + " " + text_sample], 
                is_small=False,
                calc_variance=False 
            )

            # NEW: Extract the boundary-tightening vector (LDA)
            # If the task is generative (label == -1), skip LDA and return a zero vector
            if text_label == -1:
                T_lda = torch.zeros(1024, device=device) # d_large = 1024
            else:
                # We need a small batch of prompts with matching labels to find the discriminant
                # For speed in this massive loop, we sample 4 random items of the SAME task
                lda_sample_size = min(4, len(task_config["data"]))
                lda_texts_labels = random.sample(task_config["data"], lda_sample_size)
                
                lda_prompts = [active_prompt + " " + txt for txt, lbl in lda_texts_labels]
                lda_labels = [lbl for txt, lbl in lda_texts_labels]
                
                # Append our current sample so it's included in the calculation
                lda_prompts.append(active_prompt + " " + text_sample)
                lda_labels.append(text_label)
                
                T_lda = harvester.extract_precision_targets(large_model, lda_prompts, lda_labels)

            record = {
                "task_type": chosen_task, 
                "A_small": A_small[0].cpu().to(torch.float32), 
                "B_small": B_small[0].cpu().to(torch.float32),
                "prompt_emb": random.choice(task_prompt_embs[chosen_task]).clone().to(torch.float32),
                "A_large": A_large_batch[0].cpu().to(torch.float32),
                "B_large": B_large_batch[0].cpu().to(torch.float32),
                "target_variance": task_variances[chosen_task].clone().to(torch.float32),
                "T_lda": T_lda.cpu().to(torch.float32) # NEW LINE
            }

            current_shard.append(record)

            if len(current_shard) >= shard_size:
                filepath = f"master_dataset/shard_{shard_index:04d}.pt"
                torch.save(current_shard, filepath)
                print(f"      [+] Flushed Shard {shard_index:04d} to disk ({shard_size} samples).")
                current_shard = []
                shard_index += 1

        if current_shard:
            filepath = f"master_dataset/shard_{shard_index:04d}.pt"
            torch.save(current_shard, filepath)
            print(f"      [+] Flushed final Shard {shard_index:04d} to disk.")

    print("\n[+] Massive Task-Space Extraction Complete!")

if __name__ == "__main__":
    main(num_total_samples=15000)
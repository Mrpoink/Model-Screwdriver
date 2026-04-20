import os
import random
import torch
import shutil
import time
from transformers import BertModel, BertTokenizer

from DataExtraction.TaskVectorHarvester import Harvester # Fixed import to match your filename
from DatasetBuildData import build_master_task_pool

def main(num_total_samples=50000, shard_size=2000):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Robust folder reset
    if os.path.exists("master_dataset"):
        try:
            shutil.rmtree("master_dataset")
            # Give Windows a moment to realize the folder is gone
            time.sleep(0.5) 
        except PermissionError:
            print("[!] master_dataset is locked. Cleaning contents instead...")
            for file in os.listdir("master_dataset"):
                file_path = os.path.join("master_dataset", file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"      Could not delete {file}: {e}")
    
    os.makedirs("master_dataset", exist_ok=True)
    
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
            sample_prompts = [config["prompts"][0] + " " + text for text in sample_texts]
            
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
            text_sample = task_config["data"].pop(random_idx)

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

            record = {
                "task_type": chosen_task, 
                "A_small": A_small[0].cpu().to(torch.float32), # Revert to f32 for storage/training compatibility
                "B_small": B_small[0].cpu().to(torch.float32),
                "prompt_emb": random.choice(task_prompt_embs[chosen_task]).clone().to(torch.float32),
                "A_large": A_large_batch[0].cpu().to(torch.float32),
                "B_large": B_large_batch[0].cpu().to(torch.float32),
                "target_variance": task_variances[chosen_task].clone().to(torch.float32)
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
    main(num_total_samples=500000)
import random
import torch
from datasets import load_dataset
from transformers import BertModel, BertTokenizer

from DataExtraction import TaskVectorHarvester, BuildDataset
from ScrewDriver.Tools import inject_weights, remove_weights

def prepare_imdb_data(sample_size_per_class=100) -> list:
    """Returns clean and filtered imdb dataset for task evaluation"""
    print("Downloading/Loading IMDB Dataset from HuggingFace...")
    dataset = load_dataset("imdb", split="train")
    
    short_reviews = [row for row in dataset if len(row['text'].split()) < 100]
    pos_reviews = [row['text'] for row in short_reviews if row['label'] == 1]
    neg_reviews = [row['text'] for row in short_reviews if row['label'] == 0]
    
    sampled_pos = pos_reviews[:sample_size_per_class]
    sampled_neg = neg_reviews[:sample_size_per_class]
    sampled_texts = sampled_pos + sampled_neg
    
    random.shuffle(sampled_texts)
    
    print(f"Prepared {len(sampled_texts)} total balanced IMDB reviews.")
    return sampled_texts

def prepare_agnews_data(sample_size_per_class=100) -> list:
    """Returns clean and filtered AG News dataset for topic evaluation"""
    print("Downloading/Loading AG News Dataset from HuggingFace...")
    dataset = load_dataset("ag_news", split="train")
    
    world_news = [row['text'] for row in dataset if row['label'] == 0]
    sports_news = [row['text'] for row in dataset if row['label'] == 1]
    
    sampled_world = world_news[:sample_size_per_class]
    sampled_sports = sports_news[:sample_size_per_class]
    sampled_texts = sampled_world + sampled_sports
    
    random.shuffle(sampled_texts)
    
    print(f"Prepared {len(sampled_texts)} balanced AG News articles.")
    return sampled_texts

def prepare_sst2_data(sample_size_per_class=100) -> list:
    """Returns clean and filtered GLUE SST-2 dataset for sentiment evaluation"""
    print("Downloading/Loading GLUE SST-2 Dataset from HuggingFace...")
    dataset = load_dataset("glue", "sst2", split="train")
    
    pos_reviews = [row['sentence'] for row in dataset if row['label'] == 1]
    neg_reviews = [row['sentence'] for row in dataset if row['label'] == 0]
    
    sampled_pos = pos_reviews[:sample_size_per_class]
    sampled_neg = neg_reviews[:sample_size_per_class]
    sampled_texts = sampled_pos + sampled_neg
    
    random.shuffle(sampled_texts)
    
    print(f"Prepared {len(sampled_texts)} balanced SST-2 sentences.")
    return sampled_texts

def prepare_combined_data():
    print("Loading pre-distilled Oracle datasets...")
    try:
        imdb_ds = torch.load("data_imdb_sentiment.pt", weights_only=False)
        agnews_ds = torch.load("data_ag_news_classification.pt", weights_only=False)
        sst2_ds = torch.load("data_glue_sst2.pt", weights_only=False)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the datasets. {e}")
        return

    print("Extracting tensors and combining...")
    master_records = imdb_ds.data + agnews_ds.data + sst2_ds.data
    
    print(f"Combined {len(master_records)} total records. Shuffling...")
    random.shuffle(master_records)
    
    final_dataset = BuildDataset.ScrewdriverDataset(master_records)
    torch.save(final_dataset, "data_combined.pt")
    
    print(f"\n[+] Success! Saved {len(final_dataset)} thoroughly shuffled vectors to 'data_combined.pt'")

def main(task_name="imdb_sentiment", task_label="Analyze the sentiment of this text."):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nInitializing Contrastive Task-Space Pipeline for {task_name}...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device).eval()
    harvester = TaskVectorHarvester.Harvester(small_model, large_model, tokenizer, device=device)

    # 1. DYNAMIC TASK CONFIGURATION
    if task_name == "imdb_sentiment":
        raw_texts = prepare_imdb_data(sample_size_per_class=500) 
        base_prompt = "The event occurred on a Tuesday afternoon."
        task_prompts_ensemble = [
            "Analyze the sentiment of this text:",
            "Determine if the emotion is positive or negative:",
            "Classify the feeling in this review:",
            "What is the attitude of the author?",
            "Evaluate the polarity of the sentence:",
        ]
    elif task_name == "ag_news_classification":
        raw_texts = prepare_agnews_data(sample_size_per_class=500)
        base_prompt = "The event occurred on a Tuesday afternoon."
        task_prompts_ensemble = [
            "Classify the news topic of this article:",
            "Determine the subject matter of this text:",
            "What is the primary theme of this passage?",
            "Categorize this news story:",
            "Identify the journalistic domain of this writing:",
        ]
    elif task_name == "glue_sst2":
        raw_texts = prepare_sst2_data(sample_size_per_class=500)
        base_prompt = "The event occurred on a Tuesday afternoon."
        task_prompts_ensemble = [
            "Analyze the sentiment of this text:",
            "Determine if the emotion is positive or negative:",
            "Classify the feeling in this review:",
            "What is the overall sentiment of this sentence?",
            "Is the sentiment expressed here positive or negative?",
        ]
    elif task_name == "combined":
        prepare_combined_data()
        return
    
    # --- PCA EXTRACTION REMOVED ---
    # We no longer inject a static, linear PCA matrix. We want the Screwdriver 
    # to learn the local manifold shift (Neutral -> Task Active) natively.

    # 2. GLOBAL CAUSAL TRACE (Run Once on base large model)
    print("\n[1/2] Mapping Concept Circuits (Causal Tracing)...")
    sample_prompts = [task_prompts_ensemble[0] + " " + text for text in raw_texts[:32]]
    target_variance = harvester.causal_trace_variance(large_model, sample_prompts)

    # 3. RAPID EXTRACTION LOOP
    print("\n[2/2] Starting Task-Space Dataset Generation...")
    dataset_records = []
    prompt_emb = harvester.embed_prompt(task_prompts_ensemble[0])

    for idx, text in enumerate(raw_texts):
        if idx % 50 == 0:
            print(f"      Processing sample {idx}/{len(raw_texts)}...")

        # A. Clean Input from Small Model (Ensembled)
        A_small, B_small = harvester.extract_ensembled_matrices(
            small_model, base_prompt, task_prompts_ensemble, text
        )
        
        # B. Target from Oracle Large Model (Task-Space Projection)
        # Without PCA injected, this now pure-subtracts Neutral from Active
        A_large_batch, B_large_batch, _ = harvester.extract_task_matrices(
            large_model, 
            [base_prompt + " " + text], 
            [task_prompts_ensemble[0] + " " + text], 
            is_small=False,
            calc_variance=False 
        )
        
        # C. Strip the batch dimension and append
        dataset_records.append({
            'A_small': A_small[0].cpu(),
            'B_small': B_small[0].cpu(),
            'prompt_emb': prompt_emb.cpu(),
            'A_large': A_large_batch[0].cpu(),
            'B_large': B_large_batch[0].cpu(),
            'target_variance': target_variance.cpu()
        })

    
    # 4. SAVE
    final_dataset = BuildDataset.ScrewdriverDataset(dataset_records)
    torch.save(final_dataset, f"data_{task_name}.pt")
    print(f"\n[*] Saved {len(final_dataset)} Task-Space vectors to 'data_{task_name}.pt'")
    
    del small_model, large_model, harvester
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
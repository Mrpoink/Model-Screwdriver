import random
import torch
from datasets import load_dataset
from transformers import BertModel, BertTokenizer

from DataExtraction import TaskVectorHarvester, BuildDataset

def prepare_imdb_data(sample_size_per_class=100) -> list:
    """
    Fetches the IMDB dataset, filters out long reviews to prevent truncation 
    issues with the [CLS] token, and balances positive/negative classes.
    """
    print("Downloading/Loading IMDB Dataset from HuggingFace...")
    dataset = load_dataset("imdb", split="train")
    
    # 1. Filter for brevity: < 100 words ensures fast processing and full context retention
    print("Filtering and balancing dataset classes...")
    short_reviews = [row for row in dataset if len(row['text'].split()) < 100]
    
    # 2. Segregate into positive (label 1) and negative (label 0) buckets
    pos_reviews = [row['text'] for row in short_reviews if row['label'] == 1]
    neg_reviews = [row['text'] for row in short_reviews if row['label'] == 0]
    
    # 3. Sample exactly the target amount to maintain perfect binary variance for K-Means
    sampled_pos = pos_reviews[:sample_size_per_class]
    sampled_neg = neg_reviews[:sample_size_per_class]
    
    # 4. Combine and shuffle to prevent sequential bias during extraction
    sampled_texts = sampled_pos + sampled_neg
    random.shuffle(sampled_texts)
    
    print(f"Prepared {len(sampled_texts)} total balanced IMDB reviews.")
    return sampled_texts

def main():
    # Automatically targets your CUDA-enabled GPU for hardware acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Harvester Pipeline on {device.upper()}...")

    # --- INITIALIZATION ---
    print("Loading Tokenizer and Base/Large Models...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Set models to evaluation mode (inference only, no gradients) to save VRAM
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    small_model.eval()
    
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device)
    large_model.eval()

    # Initialize harvester
    harvester = TaskVectorHarvester.Harvester(small_model, large_model, tokenizer, device=device)

    # Fetch imdb data (set for 100 for testing purposes)
    raw_imdb_texts = prepare_imdb_data(sample_size_per_class=100)
    
    # Define the behavioral steering instruction (in this case sentiment classification)
    system_instruction = "[System: Determine if the sentiment of this review is positive or negative.] "
    task_imdb_prompts = [system_instruction + text for text in raw_imdb_texts]

    # Config map for the tasks we want the Screwdriver to learn
    tasks = [
        {
            "task_label": "Analyze the sentiment of this text.",
            "base_prompts": raw_imdb_texts,
            "task_prompts": task_imdb_prompts
        }
    ]

    print("\nStarting Automated Radar and Harvesting Pipeline...")
    dataset_records = []

    for config in tasks:
        print(f"\n--- Processing Task: {config['task_label']} ---")
        
        # Find layer to monitor
        print("Scanning Small Model for optimal layer...")
        optimal_small = harvester.find_optimal_layer(small_model, config['base_prompts'], config['task_prompts'])
        
        print("Scanning Large Model for optimal layer...")
        optimal_large = harvester.find_optimal_layer(large_model, config['base_prompts'], config['task_prompts'])
        
        # Extract the A and B matrices
        print(f"Extracting matrices (Small Layer: {optimal_small}, Large Layer: {optimal_large})...")
        A_small_batch, B_small_batch = harvester.extract_task_matrices(
            small_model, optimal_small, config['base_prompts'], config['task_prompts']
        )
        A_large_batch, B_large_batch = harvester.extract_task_matrices(
            large_model, optimal_large, config['base_prompts'], config['task_prompts']
        )
        
        # Get the textual task embedding
        prompt_emb = harvester.embed_prompt(config['task_label'])
        
        # Package up for dataset
        num_samples = A_small_batch.shape[0]
        for i in range(num_samples):
            dataset_records.append({
                'A_small': A_small_batch[i],
                'B_small': B_small_batch[i],
                'prompt_emb': prompt_emb, # Constant across all samples for this task
                'small_layer': torch.tensor(optimal_small, dtype=torch.long),
                'large_layer': torch.tensor(optimal_large, dtype=torch.long),
                'A_large': A_large_batch[i],
                'B_large': B_large_batch[i]
            })

    # Save dataset
    final_dataset = BuildDataset.ScrewdriverDataset(dataset_records)
    torch.save(final_dataset, "screwdriver_training_data.pt")
    print(f"\nSuccessfully saved {len(final_dataset)} task vectors to 'screwdriver_training_data.pt'")

if __name__ == "__main__":
    main()
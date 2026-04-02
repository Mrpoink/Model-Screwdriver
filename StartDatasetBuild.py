import random
import torch
from datasets import load_dataset
from transformers import BertModel, BertTokenizer

from DataExtraction import TaskVectorHarvester, BuildDataset

def prepare_imdb_data(sample_size_per_class=100) -> list:
    """returns clean and filtered imdb dataset

    Args:
        sample_size_per_class (int, optional): Amount of points for each binary category. Defaults to 100.

    Returns:
        list: imdb dataset variation
    """
    
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

def main():
    
    system_instruction = "[System: Determine if the sentiment of this review is positive or negative.] "
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Harvester Pipeline on {device.upper()}...")

    print("Loading Tokenizer and Base/Large Models...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    small_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    small_model.eval()
    large_model = BertModel.from_pretrained('bert-large-uncased').to(device)
    large_model.eval()

    # Initialize harvester
    harvester = TaskVectorHarvester.Harvester(small_model, large_model, tokenizer, device=device)

    raw_imdb_texts = prepare_imdb_data(sample_size_per_class=500)
    task_imdb_prompts = [system_instruction + text for text in raw_imdb_texts]

    tasks = [
        {
            "task_label": "Analyze the sentiment of this text.",
            "base_prompts": raw_imdb_texts,
            "task_prompts": task_imdb_prompts
        }
    ]

    print("\nStarting Harvesting Pipeline...")
    dataset_records = []

    for config in tasks:
        print(f"\n--- Processing Task: {config['task_label']} ---")
        
        
        # Extract the A and B matrices
        print(f"Extracting matrices...")
        
        A_small_batch, B_small_batch, optimal_small_layers = harvester.extract_task_matrices(
            small_model, config['base_prompts'], config['task_prompts'], is_small=True
        )
        A_large_batch, B_large_batch, optimal_large_layers = harvester.extract_task_matrices(
            large_model, config['base_prompts'], config['task_prompts']
        )
        
        print("Found optimal layers")
        
        
        # Get the textual task embedding
        prompt_emb = harvester.embed_prompt(config['task_label'])
        
        # Package up for dataset
        num_samples = A_small_batch.shape[0]
        for i in range(num_samples):
            dataset_records.append({
                'A_small': A_small_batch[i],
                'B_small': B_small_batch[i],
                'prompt_emb': prompt_emb, # Constant across all samples for this task
                'small_layer': optimal_small_layers,
                'large_layer': optimal_large_layers,
                'A_large': A_large_batch[i],
                'B_large': B_large_batch[i]
            })

    # Save dataset
    final_dataset = BuildDataset.ScrewdriverDataset(dataset_records)
    torch.save(final_dataset, "screwdriver_training_data.pt")
    print(f"\nSuccessfully saved {len(final_dataset)} task vectors to 'screwdriver_training_data.pt'")

if __name__ == "__main__":
    main()
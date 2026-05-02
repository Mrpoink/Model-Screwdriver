import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score

def tokenize_data(example, tokenizer):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

def evaluate_accuracy(model, dataloader, device):
    """Runs a quick evaluation loop to check the current accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def warm_up_teacher(dataset_name="imdb", target_acc=0.90, max_epochs=5, batch_size=16):
    """
    Trains the large model via LoRA until it reaches a specific accuracy, 
    providing a 'Gold Signal' for the TaskHarvester.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Initializing Teacher Model Warm-up on {device}...")

    # 1. LOAD DATASET (Defaulting to IMDB for binary sentiment)
    print(f"[*] Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    # Map tokenization and format for PyTorch
    tokenized_datasets = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Subsample for faster warm-up (we don't need the whole dataset to reach 90%)
    train_subset = tokenized_datasets["train"].shuffle(seed=1068).select(range(5000))
    val_subset = tokenized_datasets["test"].shuffle(seed=1068).select(range(1000))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # 2. LOAD MODEL & LORA
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    
    # Configure LoRA to maintain a highly optimized VRAM footprint
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        # Target the exact layers the Screwdriver edits
        target_modules=["query", "value", "dense"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) # Higher LR is safe for LoRA
    
    print(f"[*] Targeting {target_acc*100}% accuracy threshold...")
    
    # 3. TRAINING LOOP
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0 and step > 0:
                print(f"      Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # 4. VALIDATION & EARLY STOPPING
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"\n[+] Epoch {epoch} Complete | Val Acc: {val_acc:.2%}\n")
        
        if val_acc >= target_acc:
            print(f"[!] Target reached. Freezing Teacher for Harvesting.")
            break
            
    # 5. MERGE AND SAVE
    print("[*] Merging LoRA adapters back into base weights...")
    # This restores the standard nn.Linear architecture for the Harvester
    model = model.merge_and_unload() 
    
    save_path = f"./warmed_up_teacher_{dataset_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[+] Teacher saved to {save_path}. Ready for StartDatasetBuild.py.")

if __name__ == "__main__":
    warm_up_teacher(dataset_name="imdb", target_acc=0.90, max_epochs=5)
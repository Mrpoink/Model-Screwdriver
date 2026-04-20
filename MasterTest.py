import time
import torch
import json
import os
from datetime import datetime
from ScrewDriver.ScrewDriverTrain import main as TrainScrewdriver
from BeginTesting import evaluate_model
from StartDatasetBuild import main as BuildDataset

def run_unified_gauntlet():
    telemetry = {
        "start_time": datetime.now().isoformat(),
        "hardware_config": "Intel i5-14400F | RTX 5060 (Train) | RTX 3060 (Data)",
        "loops": []
    }

    # Configuration for evaluation
    eval_samples = 500
    pipelines = ["imdb_sentiment", "ag_news", "combined"]
    
    for iteration in range(1, 21): # Set to 20 for full benchmark
        loop_stats = {"iteration": iteration, "pipelines": {}}
        
        for pipe in pipelines:
            print(f"\n>>> PROCESSING PIPELINE: {pipe.upper()}")
            pipe_start = time.perf_counter()
            
            # --- PHASE 1: DATASET CREATION ---
            data_start = time.perf_counter()
            # BuildDataset(num_total_samples=50000) 
            data_end = time.perf_counter()
            
            # --- PHASE 2: MODEL TRAINING ---
            train_start = time.perf_counter()
            # main() now returns (final_gen_loss, final_router_loss)
            gen_loss, router_loss = TrainScrewdriver(task_name=pipe) 
            train_end = time.perf_counter()
            
            # --- PHASE 3: EVALUATION (CROSS-CATEGORY & DOWNSTREAM) ---
            eval_start = time.perf_counter()
            # We run ALL categories against this single trained model
            # Note: evaluate_model needs to be called with initialized models
            categories = ["finance_sentiment", "tweet_emotion", "cola_grammar"]
            eval_results = {}
            
            # Initialize models once per pipeline evaluation
            # (Insert model initialization logic from BeginTesting.py here)
            
            for cat in categories:
                metrics = evaluate_model(cat_config, screwdriver, ...)
                eval_results[cat] = metrics
                pass
            
            eval_end = time.perf_counter()
            
            # --- TELEMETRY LOGGING ---
            loop_stats["pipelines"][pipe] = {
                "timings": {
                    "dataset_creation_sec": data_end - data_start,
                    "training_sec": train_end - train_start,
                    "evaluation_sec": eval_end - eval_start,
                    "total_pipeline_sec": time.perf_counter() - pipe_start
                },
                "final_residuals": {
                    "generator_mse": gen_loss,
                    "router_mse": router_loss
                },
                "downstream_metrics": eval_results
            }
            
        telemetry["loops"].append(loop_stats)
        
    # Final Save
    with open(f"Master_Telemetry_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
        json.dump(telemetry, f, indent=4)

if __name__ == "__main__":
    run_unified_gauntlet()
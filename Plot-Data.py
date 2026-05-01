import json
import matplotlib.pyplot as plt

# Load the final statistics
with open('final_results.json', 'r') as f:
    data = json.load(f)

def extract_params(version):
    loops = [p['loop'] for p in data[version]['internal_parameters']]
    betas = [p['beta'] for p in data[version]['internal_parameters']]
    lambdas = [p['restore_mag'] for p in data[version]['internal_parameters']]
    return loops, betas, lambdas

v11_loops, v11_betas, v11_lambdas = extract_params('version_11')
v12_loops, v12_betas, v12_lambdas = extract_params('version_12')

# --- Plot 1: Internal Parameters ---
fig, ax1 = plt.subplots(figsize=(10, 5))

# Beta (Left Axis)
ax1.set_xlabel('Loop Iteration (Rank)')
ax1.set_ylabel(r'Beta ($\beta$)', fontsize=12)
ax1.plot(v11_loops, v11_betas, label='v11 Beta', marker='o', color='tab:blue', linewidth=2)
ax1.plot(v12_loops, v12_betas, label='v12 Beta', marker='s', color='tab:blue', linestyle='--', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()

# Lambda (Right Axis)
ax2 = ax1.twinx()
ax2.set_ylabel(r'Lambda ($\lambda$)', fontsize=12)
ax2.plot(v11_loops, v11_lambdas, label='v11 Lambda', marker='o', color='tab:red', linewidth=2)
ax2.plot(v12_loops, v12_lambdas, label='v12 Lambda', marker='s', color='tab:red', linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:red')
lines2, labels2 = ax2.get_legend_handles_labels()

plt.title('Trajectory of Learnable Parameters (v11 vs v12)')
fig.tight_layout()
plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.savefig('internal_params.png', dpi=300)

# --- Plot 2: Benchmark Delta F1 ---
datasets = ['finance_sentiment', 'banking_intent', 'tweet_emotion', 'spam_detection']
v11_f1 = [
    data['version_11']['benchmark_statistics'].get(ds, {}).get('f1_score', {}).get('mean_improvement', 0) * 100 
    for ds in datasets
]

v12_f1 = [
    data['version_12']['benchmark_statistics'].get(ds, {}).get('f1_score', {}).get('mean_improvement', 0) * 100 
    for ds in datasets
]
plt.figure(figsize=(10, 5))
x = range(len(datasets))
plt.bar([i - 0.2 for i in x], v11_f1, 0.4, label='v11', color='navy')
plt.bar([i + 0.2 for i in x], v12_f1, 0.4, label='v12', color='skyblue')

plt.ylabel('Mean F1 Improvement (%)')
plt.xticks(x, [d.replace('_', ' ').title() for d in datasets])
plt.legend()
plt.title('Zero-Shot Mean F1 Improvement by Dataset')
plt.grid(axis='y', alpha=0.3)
plt.savefig('f1_benchmarks.png', dpi=300)
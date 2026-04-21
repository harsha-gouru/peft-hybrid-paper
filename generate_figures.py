"""Generate publication-quality figures for PEFT Hybrid CL paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

COLORS = {
    'attention': '#2d7d46',   # dark green
    'deltanet': '#c44e52',    # muted red
    'both': '#dd8452',        # muted orange
    'all_linear': '#937860',  # brown
    'base': '#4c72b0',        # steel blue
}

# Load data
with open('results_cache/cl_results.json') as f:
    cl_data = json.load(f)
with open('results_cache/final_report.json') as f:
    report = json.load(f)

# Build per-strategy seed42 runs
runs_seed42 = {}
for run in cl_data['runs']:
    if run['seed'] == 42:
        runs_seed42[run['strategy']] = run

# Summary stats
summary = {}
for s in cl_data['summary']:
    summary[s['strategy']] = s


# ===== FIGURE 1: Main result bar chart =====
fig, ax = plt.subplots(figsize=(5.5, 3.5))

strategies = ['lora_attention', 'lora_deltanet', 'lora_both', 'all_linear']
labels = ['Attention\nLoRA', 'DeltaNet\nLoRA', 'Both\nLoRA', 'All\nLinear']
colors = [COLORS['attention'], COLORS['deltanet'], COLORS['both'], COLORS['all_linear']]

means = [summary[s]['avg_final_ppl_mean'] for s in strategies]
stds = [summary[s]['avg_final_ppl_std'] for s in strategies]

bars = ax.bar(labels, means, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
ax.errorbar(range(len(strategies)), means, yerr=stds, fmt='none', ecolor='#333333',
            capsize=4, capthick=1, linewidth=1)

# Value labels
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 200,
            f'{mean:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Average Final Perplexity')
ax.set_title('Continual Learning Performance by LoRA Placement\n(Qwen3.5-0.8B, A100 80GB, 3 seeds)', fontsize=10)
ax.set_ylim(0, 13000)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

fig.savefig('figures/fig1_main_result.png')
plt.close()
print("Fig 1 done")


# ===== FIGURE 2: Forgetting trajectory (attention vs deltanet) =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.8))

domains = ['Code', 'Science', 'Conv.', 'Math']
domain_keys = ['code', 'science', 'conversation', 'math']
steps = ['Base', 'Code', 'Science', 'Conv.', 'Math']
domain_colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

for ax, strategy, title in [
    (ax1, 'lora_attention', '(a) Attention LoRA'),
    (ax2, 'lora_deltanet', '(b) DeltaNet LoRA')
]:
    run = runs_seed42[strategy]
    matrix = run['ppl_matrix']

    for i, (dkey, dname) in enumerate(zip(domain_keys, domains)):
        vals = [matrix[step][dkey] for step in range(5)]
        ax.plot(range(5), vals, marker='o', color=domain_colors[i], label=dname,
                markersize=4, linewidth=1.3)

    ax.set_xticks(range(5))
    ax.set_xticklabels(steps, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('Perplexity' if ax == ax1 else '')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

# Use log scale since values span huge range
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylim(1, 5000)
ax2.set_ylim(1, 200000)

fig.suptitle('Per-Domain Perplexity During Sequential Training (seed=42)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig2_forgetting_trajectory.png')
plt.close()
print("Fig 2 done")


# ===== FIGURE 3: MMLU comparison =====
fig, ax = plt.subplots(figsize=(5.5, 3.5))

mmlu_data = report['mmlu_results']
configs = ['base', 'all_linear_seed42', 'lora_both_seed42', 'lora_deltanet_seed42', 'lora_attention_seed42']
config_labels = ['Base\n(no adapter)', 'All\nLinear', 'Both\nLoRA', 'DeltaNet\nLoRA', 'Attention\nLoRA']
config_colors = [COLORS['base'], COLORS['all_linear'], COLORS['both'], COLORS['deltanet'], COLORS['attention']]

# Extract overall MMLU accuracy from raw_line
mmlu_accs = []
for c in configs:
    line = mmlu_data[c]['raw_line']
    # parse "|mmlu              |      2|none  |      |acc   |↑  |0.5153|±  |0.0065|"
    parts = line.split('|')
    acc = float(parts[7].strip())
    mmlu_accs.append(acc)

bars = ax.bar(config_labels, mmlu_accs, color=config_colors, width=0.6, edgecolor='white', linewidth=0.5)

# Value labels
for bar, acc in zip(bars, mmlu_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Delta labels
for i, (bar, acc) in enumerate(zip(bars, mmlu_accs)):
    if i > 0:
        delta = acc - mmlu_accs[0]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.025,
                f'{delta:+.1%}', ha='center', va='top', fontsize=7.5, color='white', fontweight='bold')

ax.set_ylabel('MMLU Accuracy (5-shot)')
ax.set_title('General Knowledge Retention After Continual Learning\n(Qwen3.5-0.8B, A100 80GB)', fontsize=10)
ax.set_ylim(0.3, 0.58)
ax.axhline(y=mmlu_accs[0], color='#999999', linestyle='--', linewidth=0.7, alpha=0.7)

fig.savefig('figures/fig3_mmlu_retention.png')
plt.close()
print("Fig 3 done")


# ===== FIGURE 4: Cross-platform normalized comparison =====
fig, ax = plt.subplots(figsize=(5.5, 3.5))

# Data from SUMMARY.md (L4 runs) + A100 paper_v1
# Attention LoRA PPL used as 1x baseline for each platform
platforms = ['Qwen3.5\n(MLX, M4 Pro)', 'Qwen3.5\n(L4, CUDA)', 'Qwen3.5\n(A100, CUDA)', 'Jamba-3B\n(L4, CUDA)']

# Normalized PPL (SSM / Attention)
deltanet_norm = [26, 17, 12, 6]
both_norm = [24, 10, 9, 4]

x = np.arange(len(platforms))
width = 0.3

bars1 = ax.bar(x - width/2, deltanet_norm, width, label='DeltaNet/SSM LoRA',
               color=COLORS['deltanet'], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, both_norm, width, label='Both LoRA',
               color=COLORS['both'], edgecolor='white', linewidth=0.5)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f'{h:.0f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(y=1, color=COLORS['attention'], linestyle='--', linewidth=1, alpha=0.8,
           label='Attention LoRA (1x baseline)')
ax.set_ylabel('Normalized PPL (Attention LoRA = 1x)')
ax.set_title('Cross-Platform Validation: Attention LoRA Advantage\nis Consistent Across Hardware and Architectures', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(platforms, fontsize=8)
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 32)

fig.savefig('figures/fig4_cross_platform.png')
plt.close()
print("Fig 4 done")


# ===== FIGURE 5: Plasticity-Stability tradeoff scatter =====
fig, ax = plt.subplots(figsize=(5, 3.5))

strategies_plot = ['lora_attention', 'lora_deltanet', 'lora_both', 'all_linear']
labels_plot = ['Attention LoRA', 'DeltaNet LoRA', 'Both LoRA', 'All Linear']
colors_plot = [COLORS['attention'], COLORS['deltanet'], COLORS['both'], COLORS['all_linear']]
markers_plot = ['o', 's', 'D', '^']

# x = CL performance (avg final PPL, log), y = MMLU accuracy
mmlu_map = {
    'lora_attention': 0.4075,
    'lora_deltanet': 0.4621,
    'lora_both': 0.4665,
    'all_linear': 0.4912,
}

for strat, label, color, marker in zip(strategies_plot, labels_plot, colors_plot, markers_plot):
    ppl = summary[strat]['avg_final_ppl_mean']
    mmlu = mmlu_map[strat]
    ax.scatter(ppl, mmlu, c=color, marker=marker, s=80, label=label, zorder=5, edgecolors='#333333', linewidth=0.5)

# Base MMLU reference
ax.axhline(y=0.5153, color='#999999', linestyle='--', linewidth=0.7, alpha=0.7)
ax.text(800, 0.518, 'Base MMLU (no adapter)', fontsize=7, color='#666666')

# Annotate the tradeoff
ax.annotate('Best CL,\nworst MMLU', xy=(761, 0.4075), xytext=(2200, 0.375),
            fontsize=7.5, color=COLORS['attention'], ha='center',
            arrowprops=dict(arrowstyle='->', color=COLORS['attention'], lw=0.8))
ax.annotate('Worst CL,\nbest MMLU', xy=(9605, 0.4912), xytext=(5500, 0.515),
            fontsize=7.5, color=COLORS['all_linear'], ha='center',
            arrowprops=dict(arrowstyle='->', color=COLORS['all_linear'], lw=0.8))

ax.set_xscale('log')
ax.set_xlabel('Average Final Perplexity (log scale, lower = better CL)')
ax.set_ylabel('MMLU Accuracy (higher = better retention)')
ax.set_title('The Plasticity-Stability Tradeoff\nby LoRA Placement Strategy', fontsize=10)
ax.legend(fontsize=8, loc='center left', framealpha=0.9)

fig.savefig('figures/fig5_tradeoff_scatter.png')
plt.close()
print("Fig 5 done")

print("\nAll figures generated in figures/")

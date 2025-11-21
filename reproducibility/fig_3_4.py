import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load from original file
in_path = Path("llm_divergence.csv")
df = pd.read_csv(in_path)

# Add raw_divergence
df["raw_divergence"] = df["llm rating"] - df["human rating"]

# Settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

bins_of_interest = [3, 2, -2, -3]
palette = sns.color_palette("Set2", len(bins_of_interest))

# --- Horizontal stacked bar chart by Model ---
model_counts = df.groupby(["model","raw_divergence"]).size().unstack(fill_value=0)
for b in bins_of_interest:
    if b not in model_counts.columns:
        model_counts[b] = 0
model_counts = model_counts[bins_of_interest]

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(model_counts))
for i, err in enumerate(bins_of_interest):
    bars = ax.barh(model_counts.index, model_counts[err], left=bottom,
                   label=str(err), color=palette[i])
    for j, bar in enumerate(bars):
        if bar.get_width() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())), ha="center", va="center", fontsize=12)
    bottom += model_counts[err].values

ax.set_xlabel("Count")
ax.set_ylabel("Model")
ax.set_title("Error Distribution by Model", pad=20)
ax.legend(title="Raw divergence", frameon=True)
plt.tight_layout()
fig_model_stacked = "error_distribution_by_model_stacked.png"
plt.savefig(fig_model_stacked, dpi=300)
plt.show()

# --- Horizontal stacked bar chart by Prompt ---
prompt_counts = df.groupby(["prompt","raw_divergence"]).size().unstack(fill_value=0)
for b in bins_of_interest:
    if b not in prompt_counts.columns:
        prompt_counts[b] = 0
prompt_counts = prompt_counts[bins_of_interest]

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(prompt_counts))
for i, err in enumerate(bins_of_interest):
    bars = ax.barh(prompt_counts.index, prompt_counts[err], left=bottom,
                   label=str(err), color=palette[i])
    for j, bar in enumerate(bars):
        if bar.get_width() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())), ha="center", va="center", fontsize=12)
    bottom += prompt_counts[err].values

ax.set_xlabel("Count")
ax.set_ylabel("Prompt")
ax.set_title("Error Distribution by Prompt", pad=20)
ax.legend(title="Raw divergence", frameon=True)
plt.tight_layout()
fig_prompt_stacked = "error_distribution_by_prompt_stacked.png"
plt.savefig(fig_prompt_stacked, dpi=300)
plt.show()

(fig_model_stacked, fig_prompt_stacked)

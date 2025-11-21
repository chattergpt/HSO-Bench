import pandas as pd
import matplotlib.pyplot as plt

# Load stats file
file_path = "All Stats.xlsx"
df_stats = pd.ExcelFile(file_path).parse("Sheet1", header=[0,1])

metrics = ["MAE","Acc","κ","r","ρ"]
prompts = [c for c in df_stats.columns.levels[0] if c not in ["Unnamed: 0_level_0","Unnamed: 1_level_0"]]

# Build tidy dataframe
frames = []
for prompt in prompts:
    sub = df_stats[["Unnamed: 0_level_0","Unnamed: 1_level_0", prompt]].copy()
    sub.columns = ["Model","Country"] + metrics
    sub["Prompt"] = prompt
    sub["Model"] = sub["Model"].ffill()
    frames.append(sub)
long_stats = pd.concat(frames, ignore_index=True)
for m in metrics:
    long_stats[m] = pd.to_numeric(long_stats[m], errors="coerce")

# Compute average correlation per country across models/prompts
country_corr = long_stats.groupby("Country")["r"].mean().reset_index()
country_corr = country_corr.sort_values("r", ascending=False)

# Highlight Algeria and Madagascar
highlight = {"Algeria": "red", "Madagascar": "red"}
colors = [highlight.get(c, "steelblue") for c in country_corr["Country"]]

# === Plot with Larger Font Sizes ===
plt.figure(figsize=(14, 8))  # larger canvas

bars = plt.bar(country_corr["Country"], country_corr["r"], color=colors)

# Font settings
label_fontsize = 20
tick_fontsize = 20
annot_fontsize = 20
title_fontsize = 20

plt.xticks(rotation=35, ha="right", fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.ylabel("Mean Correlation (r)", fontsize=label_fontsize)
plt.title("LLM–Human Alignment by Country", fontsize=title_fontsize, weight="bold")

plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

# Annotate bar values
for bar, val in zip(bars, country_corr["r"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=annot_fontsize)

plt.tight_layout()

# Save image
out_country_bar = "country_corr_barplot.png"
plt.savefig(out_country_bar, bbox_inches="tight", dpi=300, format="png")

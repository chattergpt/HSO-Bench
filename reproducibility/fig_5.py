import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load divergence file
div_path = "llm_divergence.csv"
div_div = pd.read_csv(div_path)

# Filter divergences with abs_diff = 3
div_div_big = div_div[div_div["abs_diff"]==3].copy()

# Label = identity + country
div_div_big["Label"] = (
    div_div_big["identity"] 
    + " (" + div_div_big["country"] + ")" 
    + " — " + div_div_big["model"] 
    + " / " + div_div_big["prompt"]
)

# Sort for readability
div_div_big = div_div_big.sort_values("human rating")

# Plot horizontal bars
fig, ax = plt.subplots(figsize=(9,6))
y = np.arange(len(div_div_big))
height = 0.35

ax.barh(y - height/2, div_div_big["human rating"], height, label="Human", color="steelblue")
ax.barh(y + height/2, div_div_big["llm rating"], height, label="LLM", color="salmon")

ax.set_yticks(y)
ax.set_yticklabels(div_div_big["Label"])
ax.invert_yaxis()
ax.set_xlabel("Oppression Rating (1–5)")
ax.set_xlim(0,5.5)
ax.set_title("Major Divergences (|Human – LLM| = 3)")
ax.legend(frameon=False)
ax.grid(False)  # remove grid lines

plt.tight_layout()
div_chart_horiz_path = "fig_5.png"
plt.savefig(div_chart_horiz_path, bbox_inches="tight")
plt.show()

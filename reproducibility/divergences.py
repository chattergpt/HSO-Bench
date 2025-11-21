import pandas as pd

# Load the dataset
df = pd.read_csv("merged_llm_data.csv")

# Ensure numeric columns are correctly typed
df["human rating"] = pd.to_numeric(df["human rating"], errors="coerce")
df["llm rating"] = pd.to_numeric(df["llm rating"], errors="coerce")

# Compute absolute difference
df["abs_diff"] = (df["human rating"] - df["llm rating"]).abs()

# Sort by absolute difference, largest first
df_sorted = df.sort_values("abs_diff", ascending=False)

# Save results for inspection
df_sorted.to_csv("llm_divergence_sorted.csv", index=False)

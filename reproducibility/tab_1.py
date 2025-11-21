import os
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Path to folder
folder_path = "LLM Results"

# Store all model+prompt annotated data
all_data = []

# Loop through all Excel files
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(folder_path, filename)

        # Parse model and prompt from filename
        name = os.path.splitext(filename)[0]
        parts = name.split("_")
        model = parts[0]
        prompt = "_".join(parts[1:]) if len(parts) > 1 else "vanilla"

        # Read all sheets from the Excel file
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)

            # Clean and standardize columns
            df.columns = df.columns.str.strip().str.lower()

            # Keep only valid rows
            if {"identity", "human rating", "llm rating", "llm explanation"}.issubset(df.columns):
                df = df[["identity", "human rating", "llm rating", "llm explanation"]].copy()
                df["country"] = sheet_name
                df["model"] = model
                df["prompt"] = prompt
                all_data.append(df)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)

# Drop rows with missing values
full_df = full_df.dropna(subset=["human rating", "llm rating"])
full_df["human rating"] = full_df["human rating"].astype(int)
full_df["llm rating"] = full_df["llm rating"].astype(int)

# Evaluate metrics by model and prompt
results = []
grouped = full_df.groupby(["model", "prompt"])

for (model, prompt), group in grouped:
    y_true = group["human rating"]
    y_pred = group["llm rating"]

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    results.append({
        "model": model,
        "prompt": prompt,
        "n": len(group),
        "accuracy": acc,
        "cohen_kappa": kappa,
        "mae": mae,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print and save
print(results_df.sort_values(["model", "prompt"]))
results_df.to_csv("llm_metrics_by_model_prompt.csv", index=False)

# save merged data for debugging
full_df.to_csv("merged_llm_data.csv", index=False)

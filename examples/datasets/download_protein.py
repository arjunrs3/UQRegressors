import os
import pandas as pd
import requests
from io import StringIO

# UCI Protein Structure Dataset URL (CSV format)
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"

# Download the dataset
response = requests.get(uci_url)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text))

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'protein_structure')
os.makedirs(output_dir, exist_ok=True)

# Move the target column (RMSD) to the last column
cols = [col for col in df.columns if col != "RMSD"] + ["RMSD"]
df = df[cols]

# Normalize features (zero mean, unit variance)
feature_cols = cols[:-1]
target_col = cols[-1]
df_features = df[feature_cols]
df_features = (df_features - df_features.mean()) / df_features.std()

# Normalize target by dividing by its mean
df_target = df[[target_col]] / df[target_col].mean()

# Combine normalized features and target
df_norm = pd.concat([df_features, df_target], axis=1)

# Save as CSV
output_path = os.path.join(output_dir, 'protein_structure.csv')
df_norm.to_csv(output_path, index=False)
print(f"Protein dataset saved to {output_path}")

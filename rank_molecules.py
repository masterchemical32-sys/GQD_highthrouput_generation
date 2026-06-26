import pandas as pd
import numpy as np

# ===============================
# 1. Load your dataset
# ===============================
input_file = "generated_gqds_fused_descriptors.csv"   # <-- change this
output_file = "top_50_candidates_RF_new_solar.csv"

df = pd.read_csv(input_file)

print(f"Total molecules loaded: {len(df)}")

# ===============================
# 2. Clean data (VERY IMPORTANT)
# ===============================
# Remove missing or invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[
    "BandGap_eV", 
    "SA_score", 
    "MolWt"
])

print(f"After cleaning: {len(df)} molecules")

# ===============================
# 3. Optional filtering (recommended)
# ===============================
df = df[
    (df["BandGap_eV"] > 1.8) &
    (df["BandGap_eV"] < 2.5) &
    (df["SA_score"] < 5) &    
    (df["MolWt"] > 900)
]

print(f"After filtering: {len(df)} molecules")

# ===============================
# 4. Normalize scores (robust way)
# ===============================
# Avoid scale issues by normalizing between 0–1

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

# Smaller gap = better → invert
df["gap_score"] = normalize(1 / (df["BandGap_eV"] + 1e-6))

# Lower SA = better → invert
df["sa_score"] = normalize(1 / df["SA_score"])

# Lower MolWt = better → invert
df["MolWt_score"] = normalize(-df["MolWt"])

# ===============================
# 5. Final weighted score
# ===============================
df["final_score"] = (
    0.5 * df["gap_score"] +
    0.3 * df["sa_score"] +    
    0.2 * df["MolWt_score"]
)

# ===============================
# 6. Get top candidates
# ===============================
top_candidates = df.sort_values(
    by="final_score", 
    ascending=False
).head(50)

# ===============================
# 7. Save results
# ===============================
top_candidates.to_csv(output_file, index=False)

print("Top 50 candidates saved to:", output_file)

# ===============================
# 8. Show preview
# ===============================
print(top_candidates.head())
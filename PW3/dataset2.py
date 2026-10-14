import os
import pandas as pd

# Load Dataset 2 safely using absolute/relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'chemicals_cancer.csv')

df_chem = pd.read_csv(file_path)
print("Dataset 2 successfully loaded! Shape:", df_chem.shape)

# ==========================================
# TASK 6: The Naive Analysis
# ==========================================
print("\n--- TASK 6: Naive Correlations with Malignancy ---")
naive_corrs = df_chem[['benzene', 'cadmium', 'pollution_index', 'malignancy']].corr()['malignancy']
print(naive_corrs)

# ==========================================
# TASK 7: Look Again (Controlling for Pollution)
# ==========================================
print("\n--- TASK 7: Restricted Analysis (Controlling for Confounder) ---")
# Restrict data to patients within a narrow pollution index bracket (e.g., 45 to 55)
restricted_df = df_chem[(df_chem['pollution_index'] >= 45) & (df_chem['pollution_index'] <= 55)]

print(f"Restricted dataset size: {len(restricted_df)} patients")
print(restricted_df[['benzene', 'cadmium', 'malignancy']].corr())

print(df_chem['malignancy'].value_counts(normalize=True))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# SETUP & LOADING DATA

df = pd.read_csv('Heart_disease_cleveland_new.csv')
print("Shape of dataset:", df.shape)

# SESSION 1: CHARACTERISE THE DATA
# ==========================================
print("\n--- SESSION 1 ---")
continuous_vars = df[['age', 'chol', 'trestbps', 'thalach']]
print(continuous_vars.describe())

# 1. Plotting distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
vars_to_plot = ['age', 'chol', 'trestbps', 'thalach']

for i, col in enumerate(vars_to_plot):
    ax = axes[i//2, i%2]
    sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('distributions.png')
plt.close()

# 2. Normality tests (Shapiro-Wilk)
for col in vars_to_plot:
    stat, p = stats.shapiro(df[col])
    print(f"Shapiro-Wilk for {col}: stat={stat:.4f}, p-value={p:.4e}")

# --- TASK 4: Group Comparison (thalach by target) ---
healthy = df[df['target'] == 0]['thalach']
disease = df[df['target'] == 1]['thalach']

print("--- HEALTHY GROUP (target = 0) ---")
print(f"Mean: {healthy.mean():.2f}, Std: {healthy.std():.2f}, SEM: {stats.sem(healthy):.2f}")

print("\n--- HEART DISEASE GROUP (target = 1) ---")
print(f"Mean: {disease.mean():.2f}, Std: {disease.std():.2f}, SEM: {stats.sem(disease):.2f}")

# Statistical test (Mann-Whitney U for non-normal data)
mwu_stat, mwu_p = stats.mannwhitneyu(healthy, disease)
print(f"\nMann-Whitney U test: stat={mwu_stat:.4f}, p-value={mwu_p:.4e}")

# Plotting means with uncertainty (saved as thalach_by_target.png)
plt.figure(figsize=(8, 6))
sns.barplot(x='target', y='thalach', data=df, errorbar=('ci', 68), palette='Set2', capsize=0.1)
plt.title('Maximum Heart Rate (thalach) by Health Status')
plt.xlabel('Target (0 = Healthy, 1 = Heart Disease)')
plt.ylabel('Mean Thalach (bpm)')
plt.savefig('thalach_by_target.png')
plt.close()

# --- TASK 5: Correlation (age vs thalach) ---
pearson_r, pearson_p = stats.pearsonr(df['age'], df['thalach'])
spearman_r, spearman_p = stats.spearmanr(df['age'], df['thalach'])

print(f"Pearson r: {pearson_r:.4f}, p-value={pearson_p:.4e}")
print(f"Spearman r: {spearman_r:.4f}, p-value={spearman_p:.4e}")

# Scatter plot with regression line (saved as age_vs_thalach.png)
plt.figure(figsize=(8, 6))
sns.regplot(x='age', y='thalach', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Relationship Between Age and Maximum Heart Rate (thalach)')
plt.xlabel('Age (years)')
plt.ylabel('Max Heart Rate - thalach (bpm)')
plt.savefig('age_vs_thalach.png')
plt.close()
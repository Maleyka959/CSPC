# ==========================================
# Task B1: Load and Hide the Labels (UCI Wine)
# ==========================================
from sklearn.datasets import load_wine
import pandas as pd

# Load wine dataset as a pandas DataFrame
wine = load_wine(as_frame=True)
X = wine.data          # DataFrame: 178 rows x 13 measurements[cite: 12]
cultivar = wine.target # Series of 0/1/2 -- set aside until B6[cite: 12]

# Print dataset description source and details
print("=== Dataset Description ===")
print(wine.DESCR)

# Print summary statistics to inspect ranges and standard deviations
print("\n=== X.describe() ===")
print(X.describe())

# ==========================================
# Task B2: PCA on the Raw (Unscaled) Data
# ==========================================
from sklearn.decomposition import PCA
import pandas as pd

# Fit PCA on raw X (n_components=None keeps all components)
pca_raw = PCA()
pca_raw.fit(X)

# 1. Report explained variance ratio of PC1
print(f"\n--- Explained Variance Ratio (PC1) ---")
print(f"PC1 explains: {pca_raw.explained_variance_ratio_[0]:.4f} ({pca_raw.explained_variance_ratio_[0]*100:.2f}%) of total variance")

# 2. Report sorted loadings for PC1
pc1_loadings = pd.Series(pca_raw.components_[0], index=X.columns)
print("\n--- Sorted PC1 Loadings (Raw Data) ---")
print(pc1_loadings.sort_values(ascending=False))

# ==========================================
# Task B3: Scaled PCA & Scree Plot
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Scale the data (Keep 'scaler' as we need it for B5)
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# Verify mean=0 and std=1
print("Z mean (approx 0):", np.round(Z.mean(axis=0), 5))
print("Z std (approx 1):", np.round(Z.std(axis=0), 5))

# 2. Fit a new PCA on scaled data
pca_scaled = PCA()
pca_scaled.fit(Z)

# Explained variance and cumulative sums
ev_ratio = pca_scaled.explained_variance_ratio_
cum_ratio = np.cumsum(ev_ratio)

print("\n--- Explained Variance & Cumulative Sums (Scaled PCA) ---")
for i, (ev, cum) in enumerate(zip(ev_ratio, cum_ratio), 1):
    print(f"PC{i}: Individual = {ev*100:5.2f}% | Cumulative = {cum*100:5.2f}%")

# 3. Sorted Loadings for PC1 and PC2
loadings_df = pd.DataFrame(
    pca_scaled.components_[:2].T, 
    index=X.columns, 
    columns=["PC1", "PC2"]
)
print("\n--- PC1 and PC2 Loadings ---")
print(loadings_df)

# 4. Plots: Scree Plot & PC1-PC2 Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
n_comps = len(ev_ratio)
axes[0].bar(range(1, n_comps + 1), ev_ratio, alpha=0.6, color='b', label='Individual Variance')
axes[0].step(range(1, n_comps + 1), cum_ratio, where='mid', color='r', marker='o', label='Cumulative Variance')
axes[0].axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
axes[0].set_xlabel('Principal Components')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot (Scaled PCA)')
axes[0].legend(loc='best')

# PC1-PC2 Scatter plot (unlabeled/single color)
Z_pca = pca_scaled.transform(Z)
axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1], c='purple', alpha=0.7, edgecolor='k')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Wines projected on PC1-PC2 plane')

plt.tight_layout()
plt.savefig("results/wine_pca_scaled.png")
plt.show()
print("\nPlots successfully saved to 'results/wine_pca_scaled.png'")

# ==========================================
# Task B4: K-Means Clustering Analysis
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Part (a): K=3, n_init=1 across random_states 0 to 7
print("\n--- Task B4(a): Inertia across different random_states (k=3, n_init=1) ---")
inertias_a = []
for seed in range(8):
    km_a = KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=seed)
    km_a.fit(Z)
    inertias_a.append(km_a.inertia_)
    print(f"random_state={seed}: Inertia = {km_a.inertia_:.4f}")

# Part (b): Elbow method for k = 1 to 10 with n_init=10
inertias_b = []
k_range = range(1, 11)
for k in k_range:
    km_b = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
    km_b.fit(Z)
    inertias_b.append(km_b.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias_b, marker='o', color='b', linestyle='-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method For Optimal k (Scaled Wine Data)')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/wine_kmeans_elbow.png")
plt.show()
print("\nElbow plot successfully saved to 'results/wine_kmeans_elbow.png'")

# ==========================================
# Task B5: Cluster Analysis & Real Units
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Fit final KMeans with k=3 and n_init=10
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km.fit(Z)

# Cluster sizes
cluster_counts = np.bincount(km.labels_)
print("\n--- Task B5: Cluster Sizes ---")
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}: {count} wines")

# 2. Inverse transform cluster centers back to real units
centers_real = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers_real, columns=X.columns)
print("\n--- Cluster Centers in Real Units ---")
print(centers_df)

# 3. Redraw PC1-PC2 scatter plot coloured by cluster
Z_pca = pca_scaled.transform(Z)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km.labels_, cmap='viridis', alpha=0.8, edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Wines Projected on PC1-PC2 (Colored by K-Means Clusters)')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results/wine_clusters_scatter.png")
plt.show()
print("\nCluster scatter plot successfully saved to 'results/wine_clusters_scatter.png'")

# ==========================================
# Task B6: The Reveal (Comparing 3 K-Means Versions)
# ==========================================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# (i) K-Means on Scaled Data (already computed as Z, but let's make it explicit)
km_scaled = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
labels_scaled = km_scaled.fit_predict(Z)
table_scaled = pd.crosstab(cultivar, labels_scaled)
agreement_scaled = table_scaled.max(axis=0).sum() / len(cultivar)

# (ii) K-Means on Raw, Unscaled Data (X)
km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
labels_raw = km_raw.fit_predict(X)
table_raw = pd.crosstab(cultivar, labels_raw)
agreement_raw = table_raw.max(axis=0).sum() / len(cultivar)

# (iii) K-Means on First Two Principal Components (PC1-PC2 only)
pca_2 = PCA(n_components=2)
Z_pca2 = pca_2.fit_transform(Z)
km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
labels_pca2 = km_pca2.fit_predict(Z_pca2)
table_pca2 = pd.crosstab(cultivar, labels_pca2)
agreement_pca2 = table_pca2.max(axis=0).sum() / len(cultivar)

# Combine agreement scores into a summary table
agreement_df = pd.DataFrame({
    'Model_Version': [
        'Scaled Data (13 features)', 
        'Raw Unscaled Data (13 features)', 
        'PCA PC1-PC2 (2 components)'
    ],
    'Agreement_Score': [agreement_scaled, agreement_raw, agreement_pca2]
})

print("\n--- Task B6: Agreement Scores ---")
print(agreement_df)

# Save agreement table to results/
agreement_df.to_csv("results/B_agreement.csv", index=False)
print("\nAgreement table successfully saved to 'results/B_agreement.csv'")
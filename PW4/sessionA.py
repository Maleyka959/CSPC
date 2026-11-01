"""
Session A - Task A2: Clean and look at the Concrete dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 1. Load the data from the local CSV file
df = pd.read_csv("data/concrete.csv")

# 2. Count missing values and duplicate rows
print("Missing values per column:\n", df.isna().sum())
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")

# 3. Drop duplicate rows and reset the index
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

# 4. Separate back into features (X) and target (y)
X = df.drop(columns=["strength"])
y = df["strength"]

# 5. Print descriptive statistics
print("\nFeature statistics (df.describe()):")
print(df.describe())

# 6. Plot strength against cement and age
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot: strength vs cement
ax1.scatter(X["cement"], y, alpha=0.5, color="blue")
ax1.set_xlabel("Cement (kg/m^3)")
ax1.set_ylabel("Strength (MPa)")
ax1.set_title("Strength vs Cement")
ax1.grid(True)

# Plot: strength vs age
ax2.scatter(X["age"], y, alpha=0.5, color="green")
ax2.set_xlabel("Age (days)")
ax2.set_ylabel("Strength (MPa)")
ax2.set_title("Strength vs Age")
ax2.grid(True)

plt.tight_layout()
plt.savefig("data_exploration.png")
print("\nPlot successfully saved as data_exploration.png")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the sizes of the two parts as requested by Task A3
print(f"X_train size: {X_train.shape}, y_train size: {y_train.shape}")
print(f"X_test size: {X_test.shape}, y_test size: {y_test.shape}")

# ==========================================
# Task A4: Linear Regression
# ==========================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Fit the linear regression model strictly on training data
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train, y_train)

# 2. Make predictions on train and test sets
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# 3. Calculate metrics
# RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# MAE
test_mae = mean_absolute_error(y_test, y_test_pred)

# R^2
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Adjusted R^2 for training set: R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
n = X_train.shape[0]  # number of training rows
p = X_train.shape[1]  # number of features
train_r2_adj = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

# Print results to inspect
print(f"Intercept (beta_0): {lin_reg.intercept_:.4f} MPa")
print("Coefficients (beta_1 to beta_8):")
for col, coef in zip(X_train.columns, lin_reg.coef_):
    print(f"  {col}: {coef:.4f}")

print(f"\nTrain RMSE: {train_rmse:.4f} MPa")
print(f"Test RMSE: {test_rmse:.4f} MPa")
print(f"Test MAE: {test_mae:.4f} MPa")
print(f"Train R^2: {train_r2:.4f}")
print(f"Train Adjusted R^2: {train_r2_adj:.4f}")
print(f"Test R^2: {test_r2:.4f}")

# 4. Plot residuals of the test set against age (log scale)
# Residuals = true minus predicted strength
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(X_test["age"], residuals, alpha=0.5, color="purple")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("Age (days, log scale)")
plt.ylabel("Residuals (True - Predicted Strength, MPa)")
plt.title("Linear Regression Test Residuals vs. Age")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig("linear_residuals_vs_age.png")
print("\nResidual plot saved as linear_residuals_vs_age.png")
plt.show()

# ==========================================
# Task A5: Polynomial Regression (Degrees 1 to 4)
# ==========================================
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

degrees = [1, 2, 3, 4]
poly_results = []

for d in degrees:
    # Build pipeline: Polynomial expansion -> Scaling -> Linear Regression
    poly_model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    
    # Fit strictly on training data
    poly_model.fit(X_train, y_train)
    
    # Predict on train and test
    y_train_pred = poly_model.predict(X_train)
    y_test_pred = poly_model.predict(X_test)
    
    # Metrics
    tr_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    te_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    tr_r2 = r2_score(y_train, y_train_pred)
    te_r2 = r2_score(y_test, y_test_pred)
    
    # Number of features after expansion (p)
    p_features = poly_model[0].n_output_features_
    
    # Adjusted R^2
    n = X_train.shape[0]
    tr_r2_adj = 1 - (1 - tr_r2) * (n - 1) / (n - p_features - 1)
    
    poly_results.append({
        "degree": d,
        "p": p_features,
        "train_rmse": tr_rmse,
        "test_rmse": te_rmse,
        "train_r2": tr_r2,
        "train_r2_adj": tr_r2_adj,
        "test_r2": te_r2
    })

# Print results table
print("\n--- Polynomial Regression Results ---")
print(f"{'Degree':<8}{'p':<8}{'Tr RMSE':<12}{'Te RMSE':<12}{'Tr R2':<10}{'Tr Adj R2':<12}{'Te R2'}")
for res in poly_results:
    print(f"{res['degree']:<8}{res['p']:<8}{res['train_rmse']:<12.4f}{res['test_rmse']:<12.4f}{res['train_r2']:<10.4f}{res['train_r2_adj']:<12.4f}{res['test_r2']:.4f}")

# ==========================================
# Task A6: KNN Classification
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Transform target into a binary classification problem (>= 40 MPa)
y_class = (df["strength"] >= 40).astype(int)

# Drop strength from features so the model doesn't cheat
X_class = df.drop(columns=["strength"])

# Split data with stratification to maintain class balance
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 2. Sweep K values with scaled pipeline
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_results = []

print("\n--- KNN Sweep Results ---")
for k in k_values:
    # Pipeline: StandardScaler -> KNeighborsClassifier
    knn_pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="minkowski", p=2)
    )
    knn_pipe.fit(Xc_train, c_train)
    
    tr_acc = knn_pipe.score(Xc_train, c_train)
    te_acc = knn_pipe.score(Xc_test, c_test)
    
    knn_results.append({"k": k, "train_acc": tr_acc, "test_acc": te_acc})
    print(f"K={k:<4} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

# 3. Test K=5 *without* the scaler to compare performance
unscaled_knn = KNeighborsClassifier(n_neighbors=3)
unscaled_knn.fit(Xc_train, c_train)
unscaled_unscaled_test_acc = unscaled_knn.score(Xc_test, c_test)
print(f"\nKNN (K=5) Test Accuracy *Without* Scaler: {unscaled_unscaled_test_acc:.4f}")

# 4. Repeat chosen model (e.g., K=5) across 5 different random splits
print("\n--- Stability Test Across 5 Random Splits (K=5) ---")
split_accuracies = []
for seed in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_class, y_class, test_size=0.2, random_state=seed, stratify=y_class
    )
    knn_pipe.fit(X_tr, y_tr)
    acc = knn_pipe.score(X_te, y_te)
    split_accuracies.append(acc)
    print(f"Seed {seed}: Test Accuracy = {acc:.4f}")

print(f"Spread (Max - Min) across splits: {max(split_accuracies) - min(split_accuracies):.4f}")

# Task A6: Decision Tree Classification
# ==========================================
from sklearn.tree import DecisionTreeClassifier, export_text

depths = [1, 2, 3, 4, 5, 6, 8, None]
print("\n--- Decision Tree Depth Sweep ---")
print(f"{'Depth':<8}{'Leaves':<8}{'Tr Acc':<12}{'Te Acc'}")

best_tree = None
for d in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0)
    tree.fit(Xc_train, c_train)
    
    tr_acc = tree.score(Xc_train, c_train)
    te_acc = tree.score(Xc_test, c_test)
    n_leaves = tree.get_n_leaves()
    
    print(f"{str(d):<8}{n_leaves:<8}{tr_acc:<12.4f}{te_acc:.4f}")
    if d == 3:  # Let's save a reasonable depth tree for inspection
        best_tree = tree

# Print text representation of the top levels of the tree (e.g., max_depth=2)
print("\n--- Top Levels of Decision Tree (max_depth=2) ---")
shallow_tree = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=0)
shallow_tree.fit(Xc_train, c_train)
tree_rules = export_text(shallow_tree, feature_names=list(X_class.columns))
print(tree_rules)

# ==========================================
# Task A7: Classification Metrics & Thresholds
# ==========================================
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Chosen Models: KNN (K=5 or K=7) & Decision Tree (max_depth=4)
chosen_knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
chosen_knn.fit(Xc_train, c_train)

chosen_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
chosen_tree.fit(Xc_train, c_train)

# Helper function to get metrics dict
def get_metrics(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": model_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

# Compute metrics at default threshold (0.5)
knn_preds = chosen_knn.predict(Xc_test)
tree_preds = chosen_tree.predict(Xc_test)

metrics_data = [
    get_metrics("KNN (K=5)", c_test, knn_preds),
    get_metrics("Decision Tree (depth=4)", c_test, tree_preds)
]

metrics_df = pd.DataFrame(metrics_data)
print("\n--- Task A7: Metrics Comparison (Threshold = 0.5) ---")
print(metrics_df.to_string(index=False))

# 2. Threshold Sweep for Decision Tree (0.3, 0.5, 0.7, 0.9)
tree_probs = chosen_tree.predict_proba(Xc_test)[:, 1]
thresholds = [0.3, 0.5, 0.7, 0.9]
thresh_results = []

print("\n--- Decision Tree Threshold Sweep ---")
print(f"{'Threshold':<12}{'TN':<6}{'FP':<6}{'FN':<6}{'TP':<6}{'Precision':<12}{'Recall':<12}{'F1':<12}")

for t in thresholds:
    t_preds = (tree_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, t_preds).ravel()
    prec = precision_score(c_test, t_preds, zero_division=0)
    rec = recall_score(c_test, t_preds, zero_division=0)
    f1_val = f1_score(c_test, t_preds, zero_division=0)
    
    thresh_results.append({
        "Threshold": t, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Precision": prec, "Recall": rec, "F1": f1_val
    })
    print(f"{t:<12}{tn:<6}{fp:<6}{fn:<6}{tp:<6}{prec:<12.4f}{rec:<12.4f}{f1_val:<12.4f}")

# 3. Save metrics table to results/A_metrics.csv
os.makedirs("results", exist_ok=True)
metrics_df.to_csv("results/A_metrics.csv", index=False)
print("\nMetrics successfully saved to 'results/A_metrics.csv'")

# ==========================================
# Task A8: Feature Importance Analysis
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Raw and Standardized Linear Regression coefficients (fit on X_train, y_train)
raw_lr = LinearRegression()
raw_lr.fit(X_train, y_train)
raw_coefs = raw_lr.coef_

std_lr_pipe = make_pipeline(StandardScaler(), LinearRegression())
std_lr_pipe.fit(X_train, y_train)
std_coefs = std_lr_pipe[-1].coef_

# 2. Decision Tree feature importances (from chosen tree, e.g., max_depth=4)
tree_imp = chosen_tree.feature_importances_

# 3. Permutation Importance for KNN and Decision Tree (on Xc_test, c_test with scoring="f1")
perm_knn = permutation_importance(
    chosen_knn, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0
)
perm_tree = permutation_importance(
    chosen_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0
)

# Create summary DataFrame
feature_names = X_class.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Raw Coef": raw_coefs,
    "Std Coef": std_coefs,
    "Tree Importance": tree_imp,
    "Permutation KNN (F1)": perm_knn.importances_mean,
    "Permutation Tree (F1)": perm_tree.importances_mean
})

print("\n--- Task A8: Feature Importance Table ---")
print(importance_df.to_string(index=False))

# 4. Plotting Bar Chart for selected metrics
plt.figure(figsize=(14, 6))

x = np.arange(len(feature_names))
width = 0.25

plt.bar(x - width, np.abs(importance_df["Std Coef"]), width, label="|Std Coef| (LinReg)")
plt.bar(x, importance_df["Tree Importance"], width, label="Tree Importance")
plt.bar(x + width, importance_df["Permutation Tree (F1)"], width, label="Permutation Tree (F1)")

plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Comparison of Feature Importance Metrics")
plt.xticks(x, feature_names, rotation=45)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("results/feature_importance.png")
plt.show()
print("\nPlot successfully saved to 'results/feature_importance.png'")

# ==========================================
# Bonus Task: Physics-Informed Features
# ==========================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Copy the original dataset to avoid modifying the base df
df_physics = df.copy()

# 1. Replace age with its logarithm
df_physics["age"] = np.log(df_physics["age"])

# 2. Replace water with the water-to-binder ratio (w/b)
# w/b = water / (cement + slag + ash)
df_physics["water"] = df_physics["water"] / (
    df_physics["cement"] + df_physics["slag"] + df_physics["ash"]
)
# Rename the column for clarity
df_physics = df_physics.rename(columns={"water": "wb_ratio"})

# Separate features and target
X_phys = df_physics.drop(columns=["strength"])
y_phys = df_physics["strength"]

# Train/test split using the same random state and split proportion as before
Xp_train, Xp_te, yp_train, yp_te = train_test_split(
    X_phys, y_phys, test_size=0.2, random_state=42
)

# Fit plain Linear Regression
phys_lr = LinearRegression()
phys_lr.fit(Xp_train, yp_train)

yp_pred = phys_lr.predict(Xp_te)
phys_test_r2 = r2_score(yp_te, yp_pred)

print("\n--- Bonus Task: Physics-Informed Model ---")
print(f"Number of features (p): {Xp_train.shape[1]}")
print(f"Test R²: {phys_test_r2:.4f}")
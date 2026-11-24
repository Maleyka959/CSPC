# CSPC — Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup
Create and activate the environment for a given lab:
```bash
conda env create -f PW1/Lab A/environment.yml
conda activate cspc

What I built:

    Created a radioactive decay simulation using both a pure-Python loop and a vectorized NumPy implementation, along with unit tests using pytest.

Speed comparison (loop vs NumPy):

    loop : 2.15 s

    numpy : 0.05 s

    speed-up: ~43 x faster

Tests: all passing? yes

Conclusion:

    Vectorizing calculations with NumPy drastically speeds up simulation time compared to standard Python loops.

    NumPy offloads iteration loops to optimized C-level code, avoiding the heavy overhead of Python's dynamic typing and pointer chasing.

Reproducibility (Stretch Goal):

    Partner tested: Yes

    Results: Ran successfully on partner's machine without modifications after setting up the cspc Conda environment.

## PW1 --- Lab B

### Results & Observations
- **Data Overview:** The `decay_observed.csv` dataset contains radioactive decay measurements over time, starting with an initial count of $N_0 = 5000$ at $t = 0.0$.
- **Analytical Comparison:** Comparing the scatter plot of the observed data with the analytical decay law ($N_0 e^{-\lambda t}$ with $\lambda = 0.3$) shows a strong alignment. The observed points closely follow the theoretical exponential curve, validating the model.

### Automation
- **Snakemake Pipeline:** Implemented a `Snakefile` containing a single rule (`plot`) that automatically takes `decay_observed.csv` as input, executes `plot.py`, and generates `figure.png`. Rerunning the pipeline correctly respects file timestamps, avoiding redundant computations when inputs remain unchanged.



## PW2 --- Lab A

* **Mean Acceleration:** -8.58 m/s² (with a standard deviation of 28.72 m/s²)
* **Why the acceleration was noisy:** Differentiation magnifies measurement noise; applying it twice to compute acceleration turns tiny position errors into massive fluctuations.
* **Integration results:** Integrating the noisy acceleration back up suppressed the noise, successfully recovering the trajectory with a maximum position difference of 0.78 meters.

Differentiation enhances measurement errors: the double use of this operation to calculate acceleration turns small vibrations in coordinates into huge jumps, due to which the acceleration values fluctuate strongly.

The largest difference between the original and recovered position was 0.78 meters, which is well within the expected threshold and confirms that integration successfully suppressed the noise.



### PW2 Lab B — Part 2B Results: Harder Landscape
* **Do the methods agree?** 
  No. On a complex landscape with multiple stationary points, the methods do not automatically agree. Gradient descent and SLSQP find local minima depending on the basin of attraction, whereas Newton's method simply finds roots of the derivative ($g'(x) = 0$), meaning it can land on local maxima as well as minima.
* **Did Newton land on a minimum or another stationary point?** 
  It depends on the starting point. Starting from $x_0 = 0$, Newton landed on a **local maximum** ($g''(x) < 0$). Starting from $x_0 = 2$, it landed on a **local minimum** ($g''(x) > 0$), because Newton's method targets any stationary point where the derivative is zero without checking curvature unless explicitly programmed to do so.
* **How did the starting point change the result?** 
  Because $g(x)$ has multiple local extrema, the starting point determines which "basin of attraction" the algorithm enters. A different $x_0$ can cause gradient descent and Newton to converge to completely different local minima or even a maximum.

- **Part 2 (Optimization Comparison):** Compared Gradient Descent, Newton’s method, and SLSQP on convex and complex landscapes. Newton's method converged rapidly on convex surfaces, while gradient-based methods required careful handling on complex multi-minima landscapes.
- **Part 3 (Reaction Kinetics):** Fitted the first-order rate constant $k$ using `scipy.optimize.minimize` (SLSQP), obtaining $k \approx 0.2618$ (saved as `kinetics.png`).
- **Part 4 (Chemical Equilibrium):** Solved for the reaction extent $x$ of $\text{H}_2 + \text{I}_2 \rightleftharpoons 2\text{HI}$ ($K = 15.6$) using `scipy.optimize.newton` and SLSQP. Both methods agreed at $x \approx 0.664$, yielding equilibrium amounts $\text{H}_2 = \text{I}_2 \approx 0.336\text{ mol}$ and $\text{HI} \approx 1.327\text{ mol}$ (saved as `equilibrium.png`).
- **Part 5 (Titration Equivalence Point):** Located the equivalence point at approximately $50\text{ mL}$ by finding the maximum numerical slope ($\text{dpH/dV}$) of the titration curve using `np.gradient` and `np.argmax` (saved as `titration.png`).





# Practical Work 3 — Report

## Dataset 1: Cleveland Heart Disease (`Heart_disease_cleveland_new.csv`)

### 1. Dataset Overview & Session 1: Characterising the Data
* **Shape:** 303 rows, 14 columns.
* **Continuous Variables Analyzed:** `age`, `chol`, `trestbps`, `thalach`.
* **Normality Tests (Shapiro-Wilk, alpha = 0.05):**
  * `age`: Stat = 0.9865, $p = 0.0061$ (Approximately normal with minor deviation).
  * `chol`: Stat = 0.9473, $p = 5.91 \times 10^{-9}$ (Non-normal, right-skewed).
  * `trestbps`: Stat = 0.9665, $p = 1.80 \times 10^{-6}$ (Non-normal, right-skewed).
  * `thalach`: Stat = 0.9765, $p = 7.00 \times 10^{-5}$ (Non-normal).
* **Methodological Implication:** Because key variables violate normality, non-parametric methods (Mann-Whitney U test, Spearman correlation) were selected for Session 2.

### 2. Session 2: Draw Conclusions
* **Task 4 (`thalach` vs `target`):**
  * Healthy Group (`0`): Mean = **158.38 bpm** ($\pm 1.50$ SEM).
  * Heart Disease Group (`1`): Mean = **139.26 bpm** ($\pm 1.92$ SEM).
  * **Test Result:** Mann-Whitney U test statistic = $16,989.5$, $p = 1.86 \times 10^{-13}$. 
  * **Conclusion:** Healthy patients achieve a statistically significant higher maximum heart rate than those with heart disease, confirmed visually by non-overlapping error bars in `thalach_by_target.png`.
* **Task 5 (`age` vs `thalach`):**
  * **Correlation:** Pearson $r = -0.394$ ($p = 1.11 \times 10^{-12}$).
  * **Interpretation:** Moderate negative correlation. Maximum heart rate decreases systematically as age increases (`age_vs_thalach.png`).
* **Bonus (Category Predictability):**
  * The `target` column exhibits a balanced distribution (54.1% healthy, 45.9% disease), yielding a high entropy (~0.99) and reflecting high unpredictability.

---

## Dataset 2: A Chemical-Exposure Mystery (`chemicals_cancer.csv`)

### 6. The Naive Analysis
* **Findings:** Cadmium showed a very strong naive correlation with tumour malignancy ($r = 0.88$), while benzene showed a moderate correlation ($r = 0.39$).
* **Face-Value Conclusion:** At first glance, cadmium appeared to be the primary driver of malignancy.

### 7. Confounding Analysis & True Cause
* **Method:** Controlled for potential confounding by `pollution_index` by restricting the analysis to a narrow pollution bracket ($45 \le \text{pollution\_index} \le 55$).
* **Findings:** When pollution was held constant, cadmium's correlation with malignancy completely vanished ($r = -0.04$), whereas benzene's correlation rose to $r = 0.67$. Furthermore, cadmium tracks almost perfectly with general area pollution ($r = 0.98$).
* **Conclusion:** **Benzene is the true biological cause** of malignancy. **Cadmium only looked guilty** due to confounding with area pollution.





# PW4 -- Machine Learning: Concrete Strength & Wine Clustering

## Session A: Concrete Strength (Supervised Learning)

### Task A1: Data Loading & Variables
The dataset used is the *Concrete Compressive Strength* dataset from the UCI Machine Learning Repository (ID 165), containing 1030 laboratory tests[cite: 9, 10]. 

**Variables and Units:**
- **cement**: $\text{kg/m}^3$ (component 1)
- **slag**: $\text{kg/m}^3$ (component 2)
- **ash**: $\text{kg/m}^3$ (component 3)
- **water**: $\text{kg/m}^3$ (component 4)
- **superplasticizer (superplastic)**: $\text{kg/m}^3$ (component 5)
- **coarse aggregate (coarse_agg)**: $\text{kg/m}^3$ (component 6)
- **fine aggregate (fine_agg)**: $\text{kg/m}^3$ (component 7)
- **age**: days (1 to 365)
- **strength**: $\text{MPa}$ (target variable, compressive strength)

### Task A2: Data Cleaning, Scale Analysis, and Visualization

* **Duplicate Removal and Data Leakage:**
  * *Why are duplicates problematic after splitting into train/test sets?* If an identical row (the same mix composition and age) exists in both the training and testing sets, the model can memorize the target value from the training portion. This causes data leakage, leading to an artificially inflated and overly optimistic test evaluation score.

* **Feature Scales (`df.describe()`):**
  * The statistical summary shows that features like `cement`, `coarse_agg`, and `fine_agg` have much larger numerical scales (hundreds of $\text{kg/m}^3$), whereas `superplastic` is on a much smaller scale. This indicates that scale-sensitive algorithms (such as KNN or regularized regression models) will require feature standardization (e.g., using `StandardScaler`) later on.

* **Relationship Analysis:**
  * **Strength vs Cement:** There is a clear positive trend—as the amount of cement increases, the compressive strength of the concrete (`strength`) generally increases.
  * **Strength vs Age:** Strength increases sharply in the early days following casting, after which the curve flattens out, showing that strength gains slow down over time.

### Task A4: Linear Regression Model

* **Fitted Equation:**
  strength = β₀ + β₁(cement) + β₂(slag) + β₃(ash) + β₄(water) + β₅(superplastic) + β₆(coarse_agg) + β₇(fine_agg) + β₈(age)
  *(Note: Replace β values with your script's printed coefficient outputs. Units are MPa for the intercept β₀, MPa per kg/m³ for ingredient coefficients, and MPa per day for age)*.

* **Model Performance Metrics Table:**

| Metric | Value | Unit |
| :--- | :--- | :--- |
| **Train RMSE** | *[10.0018]* | MPa |
| **Test RMSE** | *[I11.1913]* | MPa |
| **Test MAE** | *[8.8953]* | MPa |
| **Train R²** | *[0.6099]* | — |
| **Train Adjusted R²** | *[0.6059]* | — |
| **Test R²** | *[0.5802]* | — |

* **Residual Analysis (Residuals vs. Age on a Log Scale):**
  When plotting test residuals (true minus predicted) against age on a logarithmic scale, a clear systematic pattern (such as a non-random curved shape) appears. A systematic pattern in the residuals indicates that **the linear regression model fails to capture non-linear relationships** in the data—specifically, the diminishing marginal returns of curing time on concrete strength.

### Task A5: Polynomial Regression

* **Results Table:**

| Degree | Features (p) | Train RMSE | Test RMSE | Train R² | Train Adj R² | Test R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 8 | 10.0018 | 11.1913 | 0.6099 | 0.6059 | 0.5802 |
| **2** | 44 | 6.9892 | 8.3080 | 0.8095 | 0.7984 | 0.7686 |
| **3** | 164 | 4.3961 | 5.9515 | 0.9246 | 0.9053 | 0.8813 |
| **4** | 494 | 2.3339 | 39.4096 | 0.9788 | 0.9448 | -4.2061 |

* **Which degree would you choose, and why?**
  I would choose either degree 2 or degree 3. They hit the best sweet spot because they significantly improve test R² and lower RMSE compared to degree 1, successfully capturing non-linear patterns like how concrete strength changes with age.

* **What happens at degree 4, relating the number of features to the number of training rows?**
  At degree 4, the number of features ($p = 494$) gets extremely close to the number of training rows (~824). This causes severe overfitting. While the training RMSE drops to an artificially low 2.33, the test RMSE explodes to 39.41 and test R² plummets to -4.21 because the model memorizes noise instead of learning general trends.

* **Does the adjusted R² alone protect you from the problem?**
  No, adjusted R² isn't enough on its own. Even though it includes $p$ in the denominator to penalize complexity, when the feature space becomes massive, adjusted R² can behave erratically and fail to completely stop overfitting. You still need proper validation or regularization.

### Task A6: Classification Models (KNN & Decision Tree)

#### 1. K-Nearest Neighbors (KNN)

* **Sweep Results ($K \in \{1, 3, 5, 7, 15, 25, 51, 101\}$):**

| K | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- |
| **1** | 0.9975 | 0.8308 |
| **3** | 0.9341 | 0.8010 |
| **5** | 0.9067 | 0.8159 |
| **7** | 0.8930 | **0.8358** |
| **15** | 0.8483 | 0.8259 |
| **25** | 0.8408 | 0.8060 |
| **51** | 0.8047 | 0.7811 |
| **101** | 0.7575 | 0.7363 |

* **Model Choice & Justification:**
  I would choose **$K = 7$** (or $K = 5$). At $K=1$, the model overfits to the training set (train accuracy of 0.9975 vs. test accuracy of 0.8308). Increasing $K$ to 7 achieves the peak test accuracy (0.8358) by smoothing out local noise, whereas larger values ($K \ge 51$) over-smooth the decision boundaries and drop test accuracy below 78%.

* **Effect of Scaler ($K=5$):**
  * **With Scaler:** Test Accuracy = **0.8159**
  * **Without Scaler:** Test Accuracy = **0.8060**
  * *Why it matters:* KNN relies entirely on distance calculations. Without scaling, features with large numerical ranges (like `cement` in hundreds of kg/m³) completely dominate features with smaller numeric values (like `age` or `superplastic`), distorting neighbor distances.

* **Stability Test Across 5 Random Splits ($K=5$):**
  * Accuracies: `[0.7114, 0.7463, 0.7463, 0.7662, 0.7114]`
  * **Spread (Max - Min):** **0.0547** (~5.47%)
  * *Interpretation:* The spread is relatively small, showing that the model's performance is stable across different random train/test data splits and doesn't rely on a single lucky split.

---

#### 2. Decision Tree Classifier

* **Depth Sweep Results:**

| Max Depth | Leaves | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **1** | 2 | 0.7550 | 0.7015 |
| **2** | 4 | 0.7550 | 0.7015 |
| **3** | 8 | 0.8396 | 0.7960 |
| **4** | 16 | 0.8694 | 0.8259 |
| **5** | 26 | 0.8893 | 0.8458 |
| **6** | 38 | 0.9303 | 0.8408 |
| **8** | 61 | 0.9726 | 0.8557 |
| **None** | 80 | 0.9975 | **0.8806** |

* **Train vs. Test Gap & Depth Choice:**
  As the tree depth grows, the gap between training and test accuracy widens (reaching 0.9975 train accuracy when unlimited). I would choose a depth of **4 or 5** for a balance between accuracy (~84%) and generalizability, or use the unconstrained tree (`None`) if maximizing pure predictive test accuracy (0.8806) is the priority. 
  
  *What you gain/lose:* An unlimited tree gains ~3.5% test accuracy, but loses human interpretability (growing to 80 leaves) and runs a higher risk of overfitting specific noise in training.

* **Top Levels of Tree (`max_depth=2`):**
```text
|--- age <= 42.00
|   |--- cement <= 257.70 -> class: 0
|   |--- cement > 257.70  -> class: 0
|--- age > 42.00
|   |--- superplastic <= 6.45 -> class: 1
|   |--- superplastic > 6.45  -> class: 1

### Task A7: The Metrics That Matter

* **Comparison of Chosen Models (Threshold = 0.5):**

| Model | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KNN (K=5)** | 53 | 18 | 19 | 111 | 0.8159 | 0.7465 | 0.7361 | 0.7413 |
| **Decision Tree (depth=4)** | 58 | 21 | 14 | 108 | 0.8259 | 0.7342 | 0.8056 | 0.7682 |

* **Decision Tree Threshold Sweep:**

| Threshold | TN | FP | FN | TP | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.3** | 96 | 33 | 9 | 63 | 0.6562 | 0.8750 | 0.7500 |
| **0.5** | 107 | 22 | 12 | 60 | 0.7317 | 0.8333 | 0.7792 |
| **0.7** | 124 | 5 | 44 | 28 | 0.8485 | 0.3889 | 0.5333 |
| **0.9** | 126 | 3 | 46 | 26 | 0.8966 | 0.3611 | 0.5149 |

* **Construction Site Safety Analysis (FP vs. FN):**
  * **Which mistake is dangerous?** **False Positive (FP)** is dangerous. It means the model incorrectly predicts that weak concrete ($\ge 40\text{ MPa}$) is safe, leading to its use in a structure and risking potential failure. False Negatives (FN) only cause re-testing or financial waste by rejecting good concrete.
  * **Which metric controls it?** **Precision** ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) controls False Positives by ensuring that predicted positive cases are genuinely reliable.

* **Threshold Choice & Trade-offs:**
  * **Chosen Threshold:** Raising the threshold to **0.7** (or 0.9) significantly reduces False Positives (down to 5 and 3 respectively), drastically increasing Precision up to ~85–89%.
  * **What you pay for safety:** You pay with a severe drop in **Recall** (dropping to ~38% or 36%) and a rise in False Negatives, meaning many batches of strong concrete will be incorrectly rejected, increasing material waste and project delays.

### Task A8: Which Ingredients Matter? (Feature Importance)

* **Comparison of Raw vs. Standardized Coefficients:**
  * **What is the difference?** Raw coefficients have different physical units (e.g., MPa per kg/m³ for cement versus MPa per day for age) and cannot be directly compared against each other. Standardized coefficients measure the change in strength (MPa) per one standard deviation ($\sigma$) of each input, putting them on a common, comparable scale.

* **Do the methods agree on the top 2–3 most important inputs?**
  * Yes! All methods consistently highlight **`age`** (curing time) and **`cement`** as the most critical features, with **`water`** or **`superplasticizer`** following closely behind.

* **Is this ranking consistent with concrete engineering theory?**
  * Yes, completely. Concrete engineering dictates that compressive strength is controlled primarily by the **water-to-cement ratio** and the **curing age**. The models successfully capture these physical drivers.

* **Can high importance tell you that changing that ingredient would change the strength? (Causality vs. Correlation)**
  * **No (recall PW3: Correlation $\neq$ Causation).** High feature importance indicates statistical association and predictive power within the existing dataset, but it does not guarantee a direct causal effect. Confounders or hidden physical constraints could mean changing an input in practice won't yield the predicted change.

### Task A8: Which Ingredients Matter? (Feature Importance)

* **Results Table:**

| Feature | Raw Coef | Std Coef | Tree Importance | Permutation KNN (F1) | Permutation Tree (F1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cement** | 0.1167 | 12.1188 | 0.2895 | 0.1477 | 0.1932 |
| **slag** | 0.0981 | 8.4373 | 0.0199 | 0.0635 | 0.0000 |
| **ash** | 0.0846 | 5.4395 | 0.0000 | 0.0174 | 0.0000 |
| **water** | -0.1314 | -2.8279 | 0.0371 | 0.0380 | 0.0098 |
| **superplastic** | 0.3315 | 1.9708 | 0.1929 | 0.0362 | 0.1706 |
| **coarse_agg** | 0.0155 | 1.1990 | 0.0080 | 0.0294 | 0.0000 |
| **fine_agg** | 0.0206 | 1.6699 | 0.0861 | 0.0578 | 0.0663 |
| **age** | 0.1106 | 6.8945 | 0.3666 | 0.1574 | 0.1338 |

* **Comparison of Raw vs. Standardized Coefficients:**
  Raw coefficients have completely different units (e.g., MPa per kg/m³ for cement vs. MPa per day for age), making direct comparison misleading. Standardized coefficients put everything on the scale of change in strength per one standard deviation ($\sigma$), revealing that **cement** and **age** have the dominant linear effect.

* **Agreement on Top Ingredients:**
  Yes, all metrics point heavily toward **`cement`** and **`age`** as the most critical inputs, which matches concrete engineering theory where curing age and binder proportion drive compressive strength.

* **Causality vs. Correlation:**
  High feature importance shows strong statistical association in the data, but it **does not** mean changing an ingredient directly causes that strength change in practice (Correlation $\neq$ Causation), as hidden confounding factors or physical constraints may apply.

### Bonus: Let Physics Choose the Features

* **Results Summary:**
  * **Number of features ($p$):** **8** (replacing age with $\log(\text{age})$ and water with the $w/b$ ratio).
  * **Test $R^2$:** **0.8064**

* **What does this suggest about domain knowledge versus model flexibility?**
  * Domain knowledge vastly outperforms brute-force model flexibility. Instead of letting a complex algorithm blindly add polynomial terms (which caused extreme overfitting at degree 4 where $p=494$), embedding fundamental physical laws builds a compact, highly accurate model that generalizes cleanly without overfitting.

### Task B1: Load and Hide the Labels (UCI Wine Dataset)

* **Data Origin and Description:**
  The dataset contains chemical analyses of 178 wine samples from the same region in Italy, originally compiled by the Institute of Pharmaceutical and Food Analysis and Technologies in Genoa, Italy, and sourced via the UCI Machine Learning Repository[cite: 12, 13]. It includes measurements from 3 different grape cultivars (varieties), though class labels are initially hidden for unsupervised learning[cite: 12, 13].

* **Measurements ($13$ Features):**
  Includes chemical properties such as alcohol, malic acid, ash, alkaline ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, and proline[cite: 12, 13].

* **Numerical Ranges & Standard Deviations (`X.describe()`):**
  * **Largest range and standard deviation:** `proline` exhibits the largest scale, with a mean around 746.89, a standard deviation of 314.91, and a range spanning from 278 to 1680[cite: 13].
  * **Smallest range and standard deviation:** Features like nonflavanoid phenols or dilution ratios operate on much smaller numerical scales with standard deviations well below 1.0.

### Task B2: PCA on the Raw (Unscaled) Data

* **Explained Variance Ratio of PC1:**
  **0.9981** (99.81% of the total dataset variance is compressed into PC1 alone).

* **Sorted PC1 Loadings:**
  * `proline`: 0.9998
  * `magnesium`: 0.0179
  * `color_intensity`: 0.0023
  * `alcohol`: 0.0017
  * *(...and remaining features near zero or slightly negative)*

* **Which measurement dominates PC1, and why?**
  **`proline`** completely dominates PC1. As seen in Task B1 (`X.describe()`), `proline` spans a massive numerical range from 278 to 1680 with a standard deviation of ~314.9, whereas most other features range between 0 and 15 with standard deviations under 3. Because PCA maximizes variance and does *not* automatically rescale raw data, it confuses large numerical scales with high statistical importance.

* **Would you accept this PC1 as "the main chemical difference between the wines"?**
  **No.** This is an artificial mathematical artifact caused by raw units of measurement (proline values being in the hundreds/thousands while other features are single digits or decimals). It does not represent true underlying chemical variance until the features are properly standardized.

### Task B3: Scale, Then PCA Again

* **Explained Variance & Cumulative Sums:**
  * **PC1:** 36.20%
  * **PC2:** 19.21%
  * **PC1–PC2 Plane Total Variance:** **55.41%** (36.20% + 19.21%)[cite: 13]
  * **Components needed to reach 90% variance:** **8 components** (PC1 through PC8 reach **92.02%** cumulative variance)[cite: 13].

* **PC1 and PC2 Loadings Analysis:**

| Feature | PC1 Loading | PC2 Loading |
| :--- | :--- | :--- |
| **alcohol** | 0.1443 | 0.4837 |
| **malic_acid** | -0.2452 | 0.2249 |
| **ash** | -0.0021 | 0.3161 |
| **alcalinity_of_ash** | -0.2393 | -0.0106 |
| **magnesium** | 0.1420 | 0.2996 |
| **total_phenols** | 0.3947 | 0.0650 |
| **flavanoids** | 0.4229 | -0.0034 |
| **nonflavanoid_phenols** | -0.2985 | 0.0288 |
| **proanthocyanins** | 0.3134 | 0.0393 |
| **color_intensity** | -0.0886 | 0.5300 |
| **hue** | 0.2967 | -0.2792 |
| **od280/od315_of_diluted_wines** | 0.3762 | -0.1645 |
| **proline** | 0.2868 | 0.3649 |

* **Chemical Interpretation of PC Directions:**
  * **PC1 (Phenolic Content & Quality Axis):** Pushed **positive** by high phenolic compound content (`flavanoids`: +0.4229, `total_phenols`: +0.3947, dilution ratio: +0.3762, `proanthocyanins`: +0.3134)[cite: 13]. Pushed **negative** by higher acidity and astringency markers (`nonflavanoid_phenols`: -0.2985, `malic_acid`: -0.2452, `alcalinity_of_ash`: -0.2393)[cite: 13].
  * **PC2 (Intensity & Body Axis):** Pushed **positive** by color depth and richness (`color_intensity`: +0.5300, `alcohol`: +0.4837, `proline`: +0.3649, `ash`: +0.3161)[cite: 13]. Pushed **negative** by high `hue` (-0.2792)[cite: 13].

* **Visual Inspection of the Scatter Plot (`results/wine_pca_scaled.png`):**
  * By eye, the wines clearly separate into **3 distinct clusters** across the PC1–PC2 plane (one group in the upper-right, one in the lower-right, and one clearly separated cluster on the left side with low PC1).

### Task B4: K-Means Clustering

* **Part (a): Inertia Variation & `n_init`**
  * **Why do inertia values differ across `random_state` values when `n_init=1`?** K-Means relies on random initializations for starting cluster centroids. Depending on where these initial points land, the algorithm can get trapped in local minima, causing a single run (`n_init=1`) to yield varying final inertias.
  * **What does `n_init=10` do?** It executes the K-Means algorithm 10 independent times with different random centroid seeds and automatically selects the model and cluster labels corresponding to the lowest final inertia.
  * **Observed Inertia across random seeds ($k=3, n\_init=1$):**
    * `random_state=0`: 1277.9285
    * `random_state=1`: 1278.7608
    * `random_state=2`: 1277.9285
    * `random_state=3`: 1277.9285
    * `random_state=4`: 1279.9662
    * `random_state=5`: 1278.7608
    * `random_state=6`: 1282.4635
    * `random_state=7`: 1278.7608

* **Part (b): The Elbow Method & Choosing $k$**
  * **Optimal $k$:** The elbow curve (`results/wine_kmeans_elbow.png`) bends clearly at **$k = 3$**, after which the marginal reduction in inertia flattens out. This aligns perfectly with the true underlying count of 3 wine cultivars.
  * **Why can you not choose $k$ by simply minimizing inertia?** Inertia (Within-Cluster Sum of Squares) is a monotonically decreasing function of $k$. It reaches its absolute minimum of zero when $k = N$ (where every data point is its own cluster). Minimizing it blindly would always select the highest possible $k$, leading to extreme overfitting and failing to uncover meaningful groups.

### Task B5: What are the Clusters?

* **Cluster Sizes:**
  * **Cluster 0:** 65 wines
  * **Cluster 1:** 51 wines
  * **Cluster 2:** 62 wines

* **Cluster Centers in Real Units (Summary of Key Features):**
  * **Cluster 0:** Lower alcohol (~12.25%), moderate malic acid (~1.90), lower color intensity (~2.97), and moderate proline (~510.16).
  * **Cluster 1:** Higher malic acid (~3.31), lower total phenols (~1.68), lower dilution ratio (~1.70), and highest color intensity (~7.23).
  * **Cluster 2:** Highest alcohol (~13.68%), highest total phenols (~2.85), highest flavonoids, and exceptionally high proline (~1100.23).

* **Visualizing Clusters (`results/wine_clusters_scatter.png`):**
  The PC1–PC2 scatter plot colored by cluster shows clean and distinct spatial separation among the three groups, verifying that unsupervised K-Means naturally discovers the underlying cultivar structures based entirely on chemical properties.

### Task B6: The Reveal (Comparing K-Means Versions)

* **Agreement Summary Table (`results/B_agreement.csv`):**

| Model Version | Agreement Score |
| :--- | :--- |
| **Scaled Data (13 features)** | **0.9663** (96.63%) |
| **Raw Unscaled Data (13 features)** | **0.7022** (70.22%) |
| **PCA PC1–PC2 (2 components)** | **0.9663** (96.63%) |

* **Explanation of Differences:**
  * **Raw Data (0.7022):** Performs significantly worse because `proline`'s massive scale distorts Euclidean distance metrics in K-Means, obscuring all other 12 chemical dimensions[cite: 14].
  * **Scaled Data (0.9663):** Standardizing all 13 features gives equal weight to every chemical measurement, enabling K-Means to accurately match the true underlying grape varieties[cite: 14].
  * **PCA PC1–PC2 (0.9663):** Achieves the exact same peak agreement as the full 13-feature scaled dataset[cite: 14]. This proves that reducing the dataset to just its first two principal components retains virtually all of the essential cluster-separating variance while eliminating multi-dimensional noise.

* **Chemical Significance:**
  Without ever seeing the ground truth labels during training, unsupervised K-Means recovers the true grape cultivars with 96.63% accuracy[cite: 14]. This demonstrates that the three grape varieties possess distinct, highly homogeneous chemical signatures that naturally separate in feature space.

### Task B7: Automating the Lab with Snakemake

* **Reproduction Command:**
  To reproduce the entire PW4 lab workflow from scratch with a single command:
  ```bash
  snakemake --cores 1

### Bonus Task: What do labels buy you? (Supervised vs. Unsupervised)

- **Supervised Test Accuracies:**
  * **Full Scaled Features + KNN ($K=5$):** Achieved **0.9333** (93.33%) test accuracy.
  * **PCA (2 components) + KNN ($K=5$):** Achieved **0.9333** (93.33%) test accuracy.

- **Comparison with Unsupervised Agreement (from Task B6):**
  * The unsupervised K-Means agreement score was **0.9663** (96.63%).
  * Interestingly, the unsupervised K-Means agreement score slightly outperformed the supervised KNN test accuracy on this specific random split. Generally, supervised models are explicitly trained using true class labels (`cultivar`) to optimize decision boundaries, but sample splits and hyperparameter choices can cause slight variations.

- **Why is this comparison not completely fair?**
  * **Information Leakage / Supervision Advantage:** Supervised models are given direct access to the true target labels (`cultivar`) during training to guide their parameter tuning and neighborhood classification. 
  * **Unsupervised Handicap:** Unsupervised K-Means never sees the labels during training; it clusters the data purely based on geometric distance, and the "agreement score" is only computed *after the fact* by matching clusters to their most common true cultivar. Comparing a fully supervised classifier directly to an unsupervised clustering algorithm is inherently asymmetrical.


# PW5: Signal Processing — Where Was the Earthquake, and How Strong?

## Step 1: Look at the Record

* **Figures:** `step1_channels_full.png` and `step1_first_60s.png`.
* **Observations Across Channels (0–300 s):**
  * The 300-second (`30,000` samples at fs = 100 Hz) recording shows quiet background levels before roughly 110 s, followed by an initial arrival and then a much larger burst of shaking starting around 145–150 s that gradually decays.
  * The first, weaker shaking appears most prominently on the vertical **Z** channel (reaching roughly ±4 mm/s around 110 s), while it is much smaller on **N** and **E**.
  * The later, stronger shaking is largest on the horizontal channels **N** (reaching over +-12–14 mm/s) and **E** (reaching around ±15 mm/s), compared to roughly ±5–6 mm/s on **Z**.
* **First 60 s Zoom (Why the station isn't completely still):**
  * Zooming into the first 60 seconds shows continuous oscillations around ±0.005 to 0.007 mm/s with finer jitter riding on top.
  * **Why:** Even with no earthquake occurring, a sensitive broadband seismometer continuously records ambient environmental vibrations—such as ocean microseisms, wind/weather coupling, cultural/traffic activity, and minor instrumental noise.

---

## Step 2: The Spectrum — What Frequencies Are in the Record?

### 2(a) Array Index vs. Frequency (`fftfreq`)

* **Figure:** `step2a_fft_index_vs_freq.png`.
* **What `fftfreq` adds:** Raw array indices (`0` to `29,999`) are just sample positions and depend on record length N, whereas `np.fft.fftfreq(N, d=1/fs)` converts them into physical frequencies in **Hz** (spanning -50 Hz to +50 Hz for fs = 100 Hz).
* **Why the spectrum is mirrored around `0 Hz`:** Because the velocity channel `Z` is a **real-valued** signal, its Discrete Fourier Transform satisfies Hermitian symmetry (X[N-k] = X*[k]), so magnitudes are symmetric across negative and positive frequencies (with high indices near `30,000` corresponding to negative frequencies approaching `0 Hz`).

### 2(b) Log–Log Spectrum: Quiet Window vs. Full Record

* **Figure:** `step2b_loglog_spectrum.png`.
* **Background noise peak (first 60 s):** The quiet-window spectrum peaks around **0.1–0.4 Hz** (reaching above 10⁻³ mm/s), which is the microseism band driven by ocean waves interacting with coastlines.
* **Where the earthquake adds most energy:** Comparing the full record to the first 60 seconds, the earthquake elevates amplitudes across low and intermediate frequencies, with the strongest energy concentrated roughly from **~0.1 Hz up to a few Hz** (around 0.1–10 Hz), before rolling off toward higher frequencies near 50 Hz.
* **Why:** Seismic waves from an earthquake fault rupture and propagate through rock layers with finite duration and attenuation, concentrating most of their kinetic energy in low-to-moderate seismic frequencies rather than uniform white noise across all frequencies.

## Step 3: The Inverse Transform and a Manual Band-Pass Filter

### 3(a) Round-Trip Check (`fft` → `ifft`)
* **Check:** `np.allclose(Z, np.fft.ifft(np.fft.fft(Z)).real)` evaluates to `True`.
* **Why frequency-domain filtering relies on this:** Because the Fast Fourier Transform is reversible, we can transform a time-domain signal into frequency components, modify or zero out specific unwanted frequencies, and apply `ifft` to recover a clean time-domain signal without distorting the remaining frequencies.

### 3(b) Band-Pass Filter (`0.5 Hz` to `10 Hz`)
* **Justification from Step 2 spectra:** 
  * In the log–log spectrum from Step 2, the ambient background microseism noise peaks below `0.5 Hz` (around `0.1–0.4 Hz`), so setting `a = 0.5 Hz` cuts out that continuous ocean-wave background swell.
  * Most of the earthquake's seismic energy lies between `0.5 Hz` and `10 Hz`, while above `10 Hz` the signal drops into high-frequency noise, so setting `b = 10 Hz` removes high-frequency jitter while keeping the P- and S-wave arrivals sharp.

### 3(c) What Was Removed (`Z - clean`)
* **Figure:** `step3c_removed_z.png`
* **What is visible in the removed signal:** 
  * Before `110 s`, it shows the steady, slow-oscillating microseism background waves that were present in the quiet window.
  * Around `110 s` (P-wave arrival), it contains high-frequency components (`> 10 Hz`) removed from the sharp onset.
  * After `145 s`, it shows large, long-period surface waves (`< 0.5 Hz`, swinging up to around `±3 mm/s`) that arrive later in the earthquake and oscillate slowly.

---

## Step 4: The Envelope — How Strong Is the Shaking at Each Moment?

* **Figures:** `step4_envelopes_0p5s.png` and `step4_window_comparison.png`
* **Observations (`0.5 s` window):**
  * The filtered **Z** envelope (`step4_envelopes_0p5s.png`) shows two distinct bursts of shaking: a first sharp peak around `110 s` (reaching `~1.8 mm/s`) and a second sustained burst starting around `145 s` (reaching `~1.7 mm/s`).
  * The filtered **horizontal** envelope (`sqrt(N^2 + E^2)`) stays very small during the first arrival (`~0.5 mm/s` around `110 s`) and jumps dramatically during the second arrival around `145–150 s` (peaking around `8 mm/s`).
* **Window Length Comparison (`0.05 s`, `0.5 s`, and `5.0 s`):**
  * **When the window is too short (`0.05 s`):** The envelope stays very jagged and noisy (spiking above `3.2 mm/s`), following individual wave cycles rather than a smooth trend of shaking intensity.
  * **When the window is too long (`5.0 s`):** The envelope over-smooths the signal, lowering peak amplitudes (down to `~0.9–1.1 mm/s`) and smearing out the sharp onset times, making the shaking appear to start earlier than it actually does.

## Step 5: Arrival Times — The Start of the Shaking, Not Its Maximum

* **Figure:** `step5_arrivals.png`
* **P and S Arrivals:** The P-wave arrives at `t_P = 108.42 s` (5% threshold on the filtered Z envelope) and the S-wave arrives at `t_S = 144.08 s` (10% threshold after `t_P + 2 s` on the filtered horizontal envelope), giving a lag time `t_S - t_P = 35.66 s`.
* **Arrival Time vs. Maximum:** The horizontal envelope peaks at `152.03 s` (found with `find_peaks`), which is `7.95 s` after `t_S` because shaking builds up after the initial wave arrival due to rupture duration, wave interference, and scattering.
* **Why Using the Maximum Gives a Wrong Distance:** Taking `152.03 s` instead of `t_S = 144.08 s` inflates `t_S - t_P` by `7.95 s`, which would overestimate the earthquake distance by roughly `68.2 km` (`7.95 * 8.58 km`).

## Step 6: How Far Away Was the Earthquake?

* **Estimated Distance R:** Using `v_P = 6.2 km/s` and `v_S = 3.6 km/s` with `t_S - t_P = 35.66 s` gives `R = 306.13 km`.
* **Effect of 1 s Uncertainty in t_S:** A `1 s` uncertainty in `t_S` changes `R` by `+/- 8.58 km`.
* **Origin Time:** The earthquake origin time is `59.04 s` (`108.42 - 306.13 / 6.2`).

## Step 7: The Spectrogram — Which Frequencies, and When

* **Figure:** `step7_spectrogram_z.png`
* **Wave Arrivals in the Spectrogram:** The P-wave onset appears around `108 s` (`t_P = 108.42 s`), the S-wave onset appears around `144 s` (`t_S = 144.08 s`), and later, slower surface waves follow after `~150 s`.
* **Highest vs. Lowest Frequencies:** The P-wave carries higher frequencies extending across several Hz up to `10 Hz`, while the later surface waves and persistent background band (`~0.2–0.5 Hz`) have strong low-frequency energy.
* **What the Spectrogram Shows That Step 2 Cannot:** The single spectrum in Step 2 averages frequency content across time and cannot show when each arrival occurs, whereas the spectrogram shows how frequencies evolve over time.

## Step 8: The Magnitude

* **PGV and Magnitude M:** The raw horizontal peak ground velocity is `PGV = 1.510491 cm/s`, which together with `R = 306.13 km` in `M = (log10(PGV [cm/s]) + log10(R [km]) + 1.0) / 0.55` gives an estimated magnitude of `M = 6.66`.
* **Why `log10(R)` Is in the Formula:** Seismic wave amplitudes decrease with distance due to geometric spreading and attenuation; without `log10(R)`, the same earthquake recorded twice as far away would appear smaller and underestimate magnitude.
* **PGV Increase for +1 Magnitude Unit:** At the same distance, increasing `M` by `1.0` unit increases `log10(PGV)` by `0.55`, so PGV increases by a factor of `10^0.55 ≈ 3.55` times.

## Step 9: The Whole System

* **Pipeline Scripts:** `analyse.py` runs an unchanged `analyse(csv_file)` function on `data/seismogram_A.csv` and `data/seismogram_B.csv` to save `results/events.csv`, and `report.py` reads `results/events.csv` to write `results/report.txt`.
* **Which Was Closer and Stronger:** `seismogram_B` was closer (`R = 156.07 km` vs. `306.13 km` for `seismogram_A`), while `seismogram_A` was stronger (`M = 6.66` vs. `M = 5.36` for `seismogram_B`).

## Step 10: Automate It with Snakemake

* **Snakemake Workflow:** `Snakefile` defines `analyse`, `report`, and `all` with input and output file dependencies so steps run in order when needed.
* **10(a) Clean Run (`rm -rf results && snakemake --cores 1`):** Both `analyse` and `report` run because `results/events.csv` and `results/report.txt` do not exist yet.
* **10(b) Immediate Rerun:** Snakemake reports that nothing needs to be done because all output files already exist and are up to date relative to their inputs.
* **10(c) Editing Only Wording in `report.py`:** Only `report` (and `all`) reruns because `report.py` is newer than `results/report.txt`, while `results/events.csv` is still up to date.
* **10(d) Changing Cut-Off `b` in `analyse.py`:** Both `analyse` and `report` rerun because changing `analyse.py` updates `results/events.csv`, making `results/report.txt` out of date; selective dependency tracking keeps downstream results consistent without rerunning untouched steps unnecessarily.

## Bonus: Your Own Seismometer at Home (phyphox)

* **Figures:** `bonus_phone_envelope.png` and `bonus_phone_magnitude_vs_energy.png`.
* **Uneven Sampling and Interpolation:** Checking `np.diff(t)` on `Raw Data.csv` shows uneven time steps ranging from `0.002583 s` to `0.022462 s` (mean `0.002587 s`), so `np.interp` places the acceleration data onto an evenly spaced time grid before computing the FFT and applying the filter.
* **Spectrum and Gravity (0 Hz):** The raw spectrum has a `0 Hz` (DC) amplitude of `9.894 m/s^2` from constant Earth gravity, which is removed by the band-pass filter (`0.5–15 Hz`) along with slow drift and high-frequency noise so the envelope highlights transient impacts.
* **M_phone vs. log10(mgh):** For a `1.5 kg` bottle (`g = 9.81 m/s^2`) dropped from `10, 20, 40, and 80 cm` (`E = 1.47 J, 2.94 J, 5.89 J, 11.77 J`; `log10(mgh) = 0.168, 0.469, 0.770, 1.071`), the four detected peaks at `5.37 s`, `9.35 s`, `15.25 s`, and `39.66 s` give amplitudes of `0.0627 m/s^2`, `0.0826 m/s^2`, `0.0617 m/s^2`, and `0.0671 m/s^2`. Using `A1 = 0.0617 m/s^2` as the minimum peak gives `M_phone = log10(A / A1)` values of `0.007`, `0.127`, `0.000`, and `0.037` (with a slight/irregular variation across drop heights, `r = -0.085`, due to local floor coupling, impact damping, and ambient vibrations at `~1 m` distance).
* **Connection to Step 8 Magnitude:** Both `M_phone` and the Step 8 magnitude `M` use a base-10 logarithm of recorded ground motion amplitude, but `M_phone` is a relative ratio (`log10(A / A1)`) against the smallest local peak with no distance term, whereas Step 8 magnitude `M` uses `PGV` and includes `+ log10(R)` to account for amplitude decay with earthquake distance `R`.
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
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
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
# Introduction

This repository hosts the implementation for **Quantile Regression with Measurement Errors in Covariates (QR-MEC)**.

Our approach has features:
* **General Applicability:** The method is valid for both **linear and nonlinear** quantile regression models; and
* **Consistency Guaranteed:** The resulting estimator is shown to achieve the standard **root-_n_ consistency** and asymptotic normality under mild regularity conditions; and
* **Flexible Quantile Requirements:** The method does not impose the often-restrictive requirement of simultaneous quantile estimation across multiple levels.

Our approach has key estimation strategies:
* **Kernel Smoothing:** We circumvent the difficulties of discontinuity inherent in the quantile loss by employing kernel smoothing techniques.
* **Complex Domain Extension:** We overcome the measurement error problem in covariates by adding a “cancel variate” $`\sqrt{-1}`$__V__, which extends the estimating equation to the **complex domain**.

## Folder Structure
simulation_project/                 # ← root of repository
├── simulations/
│   ├── sim_1.py
│   ├── sim_2.py
│   ├── sim_3.py
│   └── sim_4.py
├── utils/
│   ├── init.py                 # important for package imports!
│   ├── plotting.py
│   ├── physics.py
│   ├── statistics.py
│   ├── io_utils.py
│   └── helpers.py
├── main.py                         # (optional) launcher script
├── README.md
├── requirements.txt
└── .gitignore



## Usage and Example

This repository includes an example to illustrate the usage on real-world data.
* The analysis of the **Cherry Blossom waiting time in Japan (2024)**.
---

# Citation

*(Will include the full citation here once our journal paper published.)*

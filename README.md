# Introduction

This repository hosts the implementation for **Quantile Regression with Measurement Errors (QR-ME)** in covariates, a novel estimator designed to address the challenge in statistical modeling.

Our approach has features:
* **General Applicability:** The method is valid for both **linear and nonlinear** quantile regression models; and
* **Consistency Guaranteed:** The resulting estimator is shown to achieve the standard **root-_n_ consistency** and asymptotic normality under mild regularity conditions; and
* **Flexible Quantile Requirements:** The method does not impose the often-restrictive requirement of simultaneous quantile estimation across multiple levels.

Our approach has key estimation strategies:
* **Kernel Smoothing:** We circumvent the difficulties of discontinuity inherent in quantile regression by employing kernel smoothing techniques.
* **Complex Domain Extension:** We overcome the measurement error problem in covariates by adding a “cancel variate” $`\sqrt{-1}`$__V__, which extends the estimating equation to the **complex domain**.

# Usage and Example

This repository includes an example to illustrate the usage on real-world data.
* The analysis of the **Cherry Blossom full bloom times in Japan (2024)**.

# Installation

*(Instructions on how to install or load the package/scripts go here. Python.)*

---

## Citation

*(Will include the full citation here once our journal paper published.)*

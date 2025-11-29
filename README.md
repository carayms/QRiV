# Introduction

This repository hosts the implementation for **Quantile Regression with Normal Measurement Errors (QR-ME)** in covariates, a novel estimator designed to address a challenge in statistical modeling.

Our approach has key features:
* **General Applicability:** The method is valid for both **linear and nonlinear** quantile regression models; and
* **Consistency Guaranteed:** The resulting estimator is shown to achieve the standard **root-_n_ consistency** and asymptotic normality under mild regularity conditions; and
* **Flexible Quantile Requirements:** Our method does not impose the often-restrictive requirement of simultaneous quantile estimation across multiple levels.

Our approach has novel estimation strategies:
* **Kernel Smoothing:** We circumvent the difficulties of discontinuity inherent in quantile regression by employing kernel smoothing techniques.
* **Complex Domain Extension:** We overcome the measurement errors problem by an extension of estimating equations to the **complex domain**, which is justified by the **Moment Generating Functions (MGFs)**.

# Examples

This repository includes an example to illustrate the usage on real-world data.
* The analysis of the **Cherry Blossom full bloom times in Japan (2024)**.

# Installation

*(Instructions on how to install or load the package/scripts go here. Python.)*

## Usage

*(Code snippets demonstrating the core function calls for estimation and replication go here.)*

---

## Citation

*(Will include the full citation when ourr journal paper here once published.)*

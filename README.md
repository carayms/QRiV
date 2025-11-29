| Title/Version | Author | Last Updated | Contact |
| :--- | :--- | :--- | :--- |
| **QRiV 1.0.0** | Mushan Li | 2025-11-29 | caray.msli@gmail.com |

# Introduction

## QRiV: Quantile Regression with Normal Measurement Errors (QR-ME)

This repository hosts the implementation for **QRiV**, a novel estimator designed to address a critical challenge in statistical modeling: **Quantile Regression (QR) in the presence of Normal Measurement Errors (ME) in covariates.**

Our approach is the first to achieve **consistent estimation** in this general QR-ME problem, bridging a long-standing gap in the literature.

## Key Features & Methodology

The QRiV package provides a robust and flexible framework, applicable to various modeling scenarios:

* **General Applicability:** The method is valid for both **linear and nonlinear** quantile regression models.
* **Consistency Guaranteed:** The resulting estimator is shown to achieve the standard **root-$n$ consistency** and asymptotic normality under mild regularity conditions.
* **Novel Estimation Strategy:**
    * **Kernel Smoothing:** We circumvent the difficulties of discontinuity inherent in quantile regression by employing kernel smoothing techniques.
    * **Complex Domain Extension:** We overcome the intrinsic nonlinearity of the problem by considering an extension to the **complex domain** and utilizing **Moment Generating Functions (MGFs)**.
* **Flexible Quantile Requirements:** Our method does not impose the often-restrictive requirement of simultaneous quantile estimation across multiple levels.

## Examples

This repository includes one example to illustrate the usage.

* **Real Data Application:** The complete analysis of the **Cherry Blossom full bloom times in Japan (2024)**, demonstrating the method’s practical performance on real-world data.

## Installation

*(Instructions on how to install or load the package/scripts go here. Python.)*

## Usage

*(Code snippets demonstrating the core function calls for estimation and replication go here.)*

---

## 📄 Citation

*(Will include the full citation when ourr journal paper here once published.)*

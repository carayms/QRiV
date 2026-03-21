# Introduction

This repository hosts the implementation for **Quantile Regression with Measurement Errors in Covariates (QR-MEC)**.

Our approach has features:
* **General Applicability:** The method is valid for both **linear and nonlinear** quantile regression models; and
* **Consistency Guaranteed:** The resulting estimator is shown to achieve the standard **root-_n_ consistency** and asymptotic normality under mild regularity conditions; and
* **Flexible Quantile Requirements:** The method does not impose the often-restrictive requirement of simultaneous quantile estimation across multiple levels.

Our approach has key estimation strategies:
* **Kernel Smoothing:** We circumvent the difficulties of discontinuity inherent in the quantile loss by employing kernel smoothing techniques.
* **Complex Domain Extension:** We overcome the measurement error problem in covariates by adding a “cancel variate” $`\sqrt{-1}`$__V__, which extends the estimating equation to the **complex domain**.
  
```
QRiV/
├── simulations/
│   ├── Simu_I_qriv.py
│   ├── Simu_I_weic.py
│   ├── Simu_I_clet.py
│   ├── Simu_II_qriv.py
│   ├── Simu_II_weic.py
│   ├── Simu_II_clet.py
│   ├── Simu_III.py
│   └── Simu_IV.py
├── utils/
│   ├── QRfuncV1.py
│   ├── WeiCarrollMethod.py
│   └── realDatFuncs.py
├── 2024JCherryBlossoms_SimuV/
│   ├── J.cherry.blossoms_2024-02.29-03.18_predi-meter.tempmean-_by.14-18.days.csv
│   ├── 2024_realDat_analysis.py
│   ├── 2024_mimic_SimuV.py
│   └── SIMEX_bdwSlec_SimuV.py
├── README.md
├── LICENSE
└── .gitignore
```

---
# Citation

*(Will include the full citation here once our journal paper published.)*

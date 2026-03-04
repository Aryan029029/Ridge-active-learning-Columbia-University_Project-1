# Ridge Regression & Active Learning (From Scratch)

This project implements Ridge Regression using the closed-form solution and an uncertainty-based sequential Active Learning strategy using pure NumPy.

##  Project Overview

The objective was to:

- Implement regularized least squares (Ridge Regression)
- Compute the weight vector using matrix algebra
- Predict outputs for test data
- Select the most informative data points using uncertainty-based active learning

All implementations were done from scratch without using sklearn.

---

##  Mathematical Concept Used

Ridge Regression Closed Form:

w = (XᵀX + λI)⁻¹ Xᵀy

Where:
- λ = Regularization parameter
- I = Identity matrix
- X = Feature matrix
- y = Target vector

Active Learning Strategy:
- Compute predictive variance
- Select index with maximum uncertainty using argmax
- Sequentially update training set

---

##  Results

- Training samples: 350
- Features: 7
- Test samples: 42
- Selected most informative indices:

[24, 15, 23, 26, 30, 31, 19, 4, 33, 14]

---

##  Tech Stack

- Python
- NumPy
- Pandas

---

##  Key Learnings

- Matrix-based machine learning implementation
- Regularization concepts
- Uncertainty quantification
- Sequential data selection
- Practical understanding of bias-variance tradeoff

---

##  Author

Aryan Sheoran  
Machine Learning & Data Science Enthusiast
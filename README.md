# Linear-vs.-Logistic-Regression-on-Adult-Dataset

#**Overview**
This repository contains the code, analysis, and visualizations for a Data Mining and Discovery assignment comparing Linear Regression and Logistic Regression on the UCI Adult Income Dataset. The project preprocesses the dataset (e.g., dropping missing values), trains Linear Regression to predict 'fnlwgt' (final census weight), and Logistic Regression for binary income classification (>50K USD). Evaluation uses MSE for Linear (quantifying prediction error) and Accuracy for Logistic (classification success). Using scikit-learn and Matplotlib, it achieves MSE ~1.2e12 for Linear and ~82% accuracy for Logistic, highlighting Linear's fit for continuous targets and Logistic's binary prowess.

**Key goals:**

Preprocess UCI Adult data (~32k samples, features like age, education, occupation).
Train/compare models: Linear for regression, Logistic for classification.
Visualize: Actual vs. predicted scatter for Linear; potential accuracy bars for Logistic.
Insights: Models reveal socioeconomic predictors (e.g., education boosts income odds).

Findings: Logistic excels at threshold decisions (e.g., 85% precision >50K); Linear captures weight variance but risks outliers. Discussed in report: Trade-offs in metrics, assumptions (linearity), and extensions (e.g., Ridge regularization).

#**Reproducing Results**

Full run: jupyter notebook notebooks/regression_analysis.ipynb (~5 min).
Splits: 80/20; no cross-val (simple baseline).
Expected: Linear MSE ~1.2e12 (high due to scale); Logistic acc ~82% (imbalanced classes).

#**Key Findings**

Linear Regression: Predicts fnlwgt with moderate fit (R²~0.15); education/age strong positive coefs. MSE low variance but absolute high (scale issue—suggest log-transform).
Logistic Regression: 82% accuracy; >50K class (24% minority) precision ~70%. Education/occupation key discriminators.
Comparison:

Model,Target,Metric,Value,Insight
Linear,fnlwgt,MSE,1.2e12,"Captures trends, outlier-sensitive"
Logistic,Income (>50K),Accuracy,0.82,Good for binary; imbalance hurts recall

Visuals: Scatter shows linearity (clustered around diagonal); accuracy bars stable train-val.
Critical Eval: Linear assumes homoscedasticity (violated—hetero residuals); Logistic needs balanced classes (SMOTE future). Both interpretable via coefs; ensemble (e.g., Random Forest) could boost +5%.

#**Data Sources**

Primary: UCI Adult (1994 Census; ~48k raw, 32k cleaned).
Features: Age, sex, race, marital-status, education-num, occupation, hours-per-week, native-country.
Targets: fnlwgt (continuous, 10^5–10^6), income (binary: <=/>50K).


#**Methods**

Preprocessing: Drop NaNs; one-hot/ordinal encoding for categoricals.
Models: scikit-learn defaults; no tuning (baseline).
Eval: MSE (regression error), Accuracy (classification % correct).
Viz: Scatter for residuals; potential ROC for Logistic.

#**Limitations and Future Work**

No tuning/cross-val; add GridSearch/Ridge for overfitting.
Imbalance in income; future: Stratified split, metrics (F1, AUC).
Scale fnlwgt; extend to full pipeline (pipeline lib).

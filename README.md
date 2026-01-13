# Credit-Scoring-Prediction-Model

# Credit Risk Modeling (Lending Club)

## Problem Statement
Develop a binary classification model to predict loan default (Good vs Bad) using historical loan application data, with a focus on ranking borrowers by credit risk under class imbalance.

## Data
Public Lending Club loan dataset containing borrower information, loan characteristics, and repayment outcomes.

## Methodology

### Exploratory Data Analysis (EDA)
- Univariate analysis to understand variable distributions
- Bivariate analysis to examine relationships between features and default behavior
- Identification of missing values, outliers, and data quality issues

### Feature Engineering & Selection
- Feature transformation and encoding
- Weight of Evidence (WOE) and Information Value (IV) analysis for predictive power assessment
- Variance Inflation Factor (VIF) analysis to address multicollinearity
- Removal of low-information and highly correlated features

### Data Partition
- Stratified train-test split to handle class imbalance

### Modeling
- Logistic Regression as a baseline and interpretable credit risk model
- Cost-sensitive learning to improve minority class (Bad) detection
- Regularization to control model complexity

### Model Tuning
- Hyperparameter tuning (regularization strength, class weights)
- Probability threshold optimization based on business-oriented metrics

### Model Evaluation
- ROC-AUC and Gini coefficient for ranking performance
- Precision, recall, and F1-score analysis with emphasis on Bad loans
- Stability comparison between training and test sets

### Scoring & Business Interpretation
- Credit score generation based on predicted default probabilities
- Score grading to segment borrowers into risk bands
- Analysis of score impact and high-level business recommendations for credit decisioning

## Results
- ROC-AUC: ~0.73 on test set
- The model demonstrates reasonable ranking power under class imbalance
- Logistic regression provides interpretable risk drivers suitable for credit scoring use cases

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

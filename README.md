# Credit Card Default Prediction

## Project Title
**Credit Card Default Prediction System**[cite: 1]

## Objective
The primary objective of this project is to develop and evaluate supervised machine learning classification models to predict whether a credit card client will default on their credit card payments in the upcoming month based on historical payment records, billing statements, and demographic features[cite: 1].

## Dataset Used
* **Dataset Name:** Default of Credit Card Clients Dataset[cite: 1]
* **Source:** UCI Machine Learning Repository[cite: 1]
* **Link:** [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)[cite: 1]
* **Dataset Overview:** Contains 30,000 instances and 24 explanatory variables, including credit limit, gender, education, marital status, age, repayment status history (from April to September), statement balances, and past payment amounts.

## Models Used
1. **Logistic Regression:**
   * *Rationale:* Used as an initial baseline classification model to establish linear decision boundaries and assess feature coefficients for binary default prediction[cite: 1].
2. **Random Forest Classifier:**
   * *Rationale:* Applied to capture non-linear relationships, handle interactions between financial variables, and mitigate overfitting via ensemble bagging and feature subsampling[cite: 1].
3. **Gradient Boosting Classifier:**
   * *Rationale:* Implemented to sequentially optimize predictive performance by minimizing pseudo-residuals and focusing on hard-to-classify default cases[cite: 1].

## Key Results
The dataset was split using an 80/20 stratified train-test split, followed by standard scaling on numerical features.

| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8090 | 0.6923 | 0.2374 | 0.3535 | 0.7180 |
| **Random Forest** | 0.8143 | 0.6380 | 0.3602 | 0.4607 | 0.7612 |
| **Gradient Boosting** | **0.8203** | **0.6720** | **0.3632** | **0.4716** | **0.7785** |

### Best Performing Model & Outcome
* **Best Model:** **Gradient Boosting Classifier** delivered the highest performance across overall accuracy (82.03%), F1 Score (0.4716), and ROC-AUC score (0.7785)[cite: 1].
* **Outcome:** Historical repayment status (specifically recent delay flags from `PAY_0` and `PAY_2`) showed the strongest predictive power in identifying credit default risk, enabling early intervention strategies for risk mitigation[cite: 1].

## Repository File Structure
* `Credit_Card_Default_Prediction.ipynb` – Complete Google Colab / Jupyter Notebook with end-to-end data preprocessing, EDA, model training, and performance evaluation[cite: 1].
* `dataset.csv` – Extracted raw dataset file used for model training[cite: 1].
* `credit_default_model.pkl` – Serialized best-performing Gradient Boosting model saved using `joblib`[cite: 1].
* `README.md` – Project overview, methodology, and evaluation summary[cite: 1].

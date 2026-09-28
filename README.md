# Bank Credit Risk Prediction using Machine Learning

##  Project Overview 

This project develops a machine learning-based credit risk classification system using banking and CIBIL-related customer information.

The objective is to classify customers into two target classes and compare the performance of two machine learning models:

- Random Forest
- Logistic Regression

The project also uses SHAP explainability and Random Forest feature importance to understand which customer attributes contribute most to model predictions.

---

##  Problem Statement

Credit risk assessment is an important task in banking. Traditional assessment methods can be enhanced using machine learning models that identify patterns in customer and credit-related information.

This project explores whether customer financial and demographic features can be used to classify credit risk and compares different classification approaches.

---

##  Dataset

The project uses the **Leading Indian Bank & CIBIL Real-World Dataset** from Kaggle.

Original dataset:
- 51,336 customer records
- 130 features

For this project, a reduced dataset containing selected relevant variables was used.

### Selected Features

- `tot_enq` — Total enquiries
- `enq_L3m` — Enquiries in the last 3 months
- `MARITALSTATUS` — Marital status
- `EDUCATION` — Education level
- `AGE` — Customer age
- `GENDER` — Gender
- `NETMONTHLYINCOME` — Net monthly income
- `Time_With_Curr_Empr` — Time with current employer
- `CC_utilization` — Credit card utilization
- `max_unsec_exposure_inPct` — Maximum unsecured exposure percentage
- `Credit_Score` — Credit score

### Target

`Target` is the classification variable used to distinguish between the two customer classes.

---

##  Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Excel

---

##  Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Feature & Target Separation
   ↓
Categorical Feature Encoding
   ↓
Stratified Train-Test Split
   ↓
5-Fold Cross-Validation
   ↓
Random Forest
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
SHAP Explainability
   ↓
Model Comparison
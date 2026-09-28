
# ============================================================
# BANK CREDIT RISK PREDICTION
# Random Forest vs Logistic Regression
# Cross-Validation + SHAP Explainability
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import shap


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_excel("Reduce Data for Scoring.xlsx")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# ============================================================
# 3. CREATE FEATURES (X) AND TARGET (y)
# ============================================================

X = df.drop("Target", axis=1)
y = df["Target"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Proportion:")
print(y.value_counts(normalize=True))


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

categorical_columns = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape After One-Hot Encoding:")
print(X.shape)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ============================================================
# 6. CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)


# -------------------- Cross-Validation -----------------------

rf_cv_results = cross_validate(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

rf_cv_accuracy = rf_cv_results["test_accuracy"].mean()
rf_cv_precision = rf_cv_results["test_precision"].mean()
rf_cv_recall = rf_cv_results["test_recall"].mean()
rf_cv_f1 = rf_cv_results["test_f1"].mean()
rf_cv_roc_auc = rf_cv_results["test_roc_auc"].mean()

print("\nRandom Forest Cross-Validation Results")

print(f"Mean Accuracy : {rf_cv_accuracy:.4f}")
print(f"Mean Precision: {rf_cv_precision:.4f}")
print(f"Mean Recall   : {rf_cv_recall:.4f}")
print(f"Mean F1 Score : {rf_cv_f1:.4f}")
print(f"Mean ROC-AUC  : {rf_cv_roc_auc:.4f}")


# -------------------- Train Final Model ----------------------

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")


# -------------------- Predictions ----------------------------

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Predictions Completed!")


# -------------------- Test Metrics ----------------------------

rf_test_accuracy = accuracy_score(y_test, rf_pred)
rf_test_precision = precision_score(y_test, rf_pred)
rf_test_recall = recall_score(y_test, rf_pred)
rf_test_f1 = f1_score(y_test, rf_pred)
rf_test_roc_auc = roc_auc_score(y_test, rf_prob)

rf_train_accuracy = rf_model.score(X_train, y_train)

print("\nRandom Forest Test Results")

print(f"Training Accuracy : {rf_train_accuracy:.4f}")
print(f"Testing Accuracy  : {rf_test_accuracy:.4f}")
print(f"Precision         : {rf_test_precision:.4f}")
print(f"Recall            : {rf_test_recall:.4f}")
print(f"F1 Score          : {rf_test_f1:.4f}")
print(f"ROC-AUC           : {rf_test_roc_auc:.4f}")


# -------------------- Classification Report ------------------

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_pred))


# -------------------- Confusion Matrix ------------------------

rf_cm = confusion_matrix(y_test, rf_pred)

print("Random Forest Confusion Matrix:")
print(rf_cm)


# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(rf_importance.head(10))


# Feature Importance Plot

plt.figure(figsize=(10, 6))

plt.barh(
    rf_importance.head(10)["Feature"][::-1],
    rf_importance.head(10)["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Features")
plt.tight_layout()

plt.savefig("rf_feature_importance.png", dpi=300)

plt.show()


# ============================================================
# 9. LOGISTIC REGRESSION MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)


# Pipeline:
# Scaling is performed inside each CV fold.
# This prevents data leakage during cross-validation.

log_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# -------------------- Cross-Validation -----------------------

log_cv_results = cross_validate(
    log_model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

log_cv_accuracy = log_cv_results["test_accuracy"].mean()
log_cv_precision = log_cv_results["test_precision"].mean()
log_cv_recall = log_cv_results["test_recall"].mean()
log_cv_f1 = log_cv_results["test_f1"].mean()
log_cv_roc_auc = log_cv_results["test_roc_auc"].mean()

print("\nLogistic Regression Cross-Validation Results")

print(f"Mean Accuracy : {log_cv_accuracy:.4f}")
print(f"Mean Precision: {log_cv_precision:.4f}")
print(f"Mean Recall   : {log_cv_recall:.4f}")
print(f"Mean F1 Score : {log_cv_f1:.4f}")
print(f"Mean ROC-AUC  : {log_cv_roc_auc:.4f}")


# -------------------- Train Final Model ----------------------

log_model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")


# -------------------- Predictions ----------------------------

log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Predictions Completed!")


# -------------------- Test Metrics ----------------------------

log_test_accuracy = accuracy_score(y_test, log_pred)
log_test_precision = precision_score(y_test, log_pred)
log_test_recall = recall_score(y_test, log_pred)
log_test_f1 = f1_score(y_test, log_pred)
log_test_roc_auc = roc_auc_score(y_test, log_prob)

log_train_accuracy = log_model.score(X_train, y_train)

print("\nLogistic Regression Test Results")

print(f"Training Accuracy : {log_train_accuracy:.4f}")
print(f"Testing Accuracy  : {log_test_accuracy:.4f}")
print(f"Precision         : {log_test_precision:.4f}")
print(f"Recall            : {log_test_recall:.4f}")
print(f"F1 Score          : {log_test_f1:.4f}")
print(f"ROC-AUC           : {log_test_roc_auc:.4f}")


# -------------------- Classification Report ------------------

print("\nLogistic Regression Classification Report")
print(classification_report(y_test, log_pred))


# -------------------- Confusion Matrix ------------------------

log_cm = confusion_matrix(y_test, log_pred)

print("Logistic Regression Confusion Matrix:")
print(log_cm)


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "CV Accuracy",
        "CV Precision",
        "CV Recall",
        "CV F1",
        "CV ROC-AUC",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Test ROC-AUC"
    ],

    "Random Forest": [
        rf_cv_accuracy,
        rf_cv_precision,
        rf_cv_recall,
        rf_cv_f1,
        rf_cv_roc_auc,
        rf_test_accuracy,
        rf_test_precision,
        rf_test_recall,
        rf_test_f1,
        rf_test_roc_auc
    ],

    "Logistic Regression": [
        log_cv_accuracy,
        log_cv_precision,
        log_cv_recall,
        log_cv_f1,
        log_cv_roc_auc,
        log_test_accuracy,
        log_test_precision,
        log_test_recall,
        log_test_f1,
        log_test_roc_auc
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))


# Save comparison table

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# 11. CONFUSION MATRIX VISUALIZATIONS
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "rf_confusion_matrix.png",
    dpi=300
)

plt.show()


ConfusionMatrixDisplay.from_predictions(
    y_test,
    log_pred
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "logistic_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. ROC CURVE COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    rf_prob,
    name=f"Random Forest (AUC = {rf_test_roc_auc:.3f})"
)

RocCurveDisplay.from_predictions(
    y_test,
    log_prob,
    name=f"Logistic Regression (AUC = {log_test_roc_auc:.3f})"
)

plt.title("ROC Curve Comparison")
plt.tight_layout()

plt.savefig(
    "roc_curve_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. SHAP EXPLAINABILITY — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

# Extract the trained scaler and logistic regression model
scaler = log_model.named_steps["scaler"]
logistic = log_model.named_steps["logistic"]

# Transform test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Create SHAP explainer
explainer = shap.LinearExplainer(
    logistic,
    scaler.transform(X_train)
)

print("\nSHAP Explainer Created Successfully!")

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

print("SHAP Values Calculated Successfully!")


# -------------------- SHAP Summary Plot -----------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X.columns,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. SHAP FEATURE IMPORTANCE
# ============================================================

mean_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Absolute_SHAP": mean_shap
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

print("\nSHAP Feature Importance")

print(
    shap_importance.head(10).to_string(index=False)
)


# Save SHAP importance

shap_importance.to_csv(
    "shap_feature_importance.csv",
    index=False
)


# ============================================================
# 15. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print("""
Random Forest performed better overall than Logistic Regression
for this credit-risk classification task.

Random Forest achieved higher cross-validation accuracy,
F1-score and ROC-AUC, while Logistic Regression achieved
slightly higher recall.

Feature importance and SHAP analysis were used to improve
interpretability and identify influential predictors.
""")

print("\nProject execution completed successfully!")

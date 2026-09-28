import pandas as pd

df = pd.read_excel("Reduce Data for Scoring.xlsx")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

# STEP 2 : Create Features (X) and Target (y)

X = df.drop("Target", axis=1)

y = df["Target"]

print("\nFeature Shape :", X.shape)
print("Target Shape :", y.shape)
# One-Hot Encoding

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

print("\nShape After Encoding")
print(X.shape)  
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# CROSS-VALIDATION
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc" # This measures how well the model can distinguish btw good customer vs bad customer 
    ],
    n_jobs=-1  #Use all available CPU cores
)

print("\nRandom Forest Cross-Validation Results")

print("Mean Accuracy:",
      cv_results["test_accuracy"].mean())

print("Mean Precision:",
      cv_results["test_precision"].mean())

print("Mean Recall:",
      cv_results["test_recall"].mean())

print("Mean F1 Score:",
      cv_results["test_f1"].mean())

print("Mean ROC-AUC:",
      cv_results["test_roc_auc"].mean())


# FINAL MODEL TRAINING
model.fit(
    X_train,
    y_train
)

print("\nRandom Forest Model Trained Successfully!")

# STEP 6 : Make Predictions

y_pred = model.predict(X_test)

print("\nPrediction Completed!")
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy :", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)
from sklearn.metrics import classification_report

print("\nClassification Report")

print(classification_report(y_test, y_pred))
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nTraining Accuracy :", train_accuracy)
print("Testing Accuracy :", test_accuracy)

                                           # LOGISTIC REGRESSION MODEL


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nData Scaling Completed!") 

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

print("\nLogistic Regression Model Created Successfully!")

# LOGISTIC REGRESSION CROSS-VALIDATION

from sklearn.model_selection import cross_validate

cv_results_log = cross_validate( 
    log_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    n_jobs=-1 
)

print("\nLogistic Regression Cross-Validation Results")

print("Mean Accuracy:",
      cv_results_log["test_accuracy"].mean())

print("Mean Precision:",
      cv_results_log["test_precision"].mean())

print("Mean Recall:",
      cv_results_log["test_recall"].mean())

print("Mean F1 Score:",
      cv_results_log["test_f1"].mean())

print("Mean ROC-AUC:",
      cv_results_log["test_roc_auc"].mean())

# TRAIN FINAL LOGISTIC REGRESSION MODEL

log_model.fit(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression Model Trained Successfully!")
# LOGISTIC REGRESSION PREDICTIONS

y_pred_log = log_model.predict(X_test_scaled)

print("\nLogistic Regression Predictions Completed!")

# LOGISTIC REGRESSION ACCURACY

from sklearn.metrics import accuracy_score

log_accuracy = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Test Accuracy :", log_accuracy)
# LOGISTIC REGRESSION CONFUSION MATRIX

from sklearn.metrics import confusion_matrix

cm_log = confusion_matrix(y_test, y_pred_log)

print("\nLogistic Regression Confusion Matrix")
print(cm_log) 

# LOGISTIC REGRESSION CLASSIFICATION REPORT

from sklearn.metrics import classification_report

print("\nLogistic Regression Classification Report")

print(classification_report(y_test, y_pred_log))
# LOGISTIC REGRESSION TRAINING VS TESTING ACCURACY

log_train_accuracy = log_model.score(
    X_train_scaled,
    y_train
)

log_test_accuracy = log_model.score(
    X_test_scaled,
    y_test
)

print("\nLogistic Regression Training Accuracy :", log_train_accuracy)
print("Logistic Regression Testing Accuracy :", log_test_accuracy)


# SHAP EXPLAINABILITY
import shap
explainer = shap.LinearExplainer(
    log_model,
    X_train_scaled
)

print("\nSHAP Explainer Created Successfully!")
shap_values = explainer.shap_values(X_test_scaled)

print("\nSHAP Values Calculated Successfully!")
shap.summary_plot(    #summary plot Gives a global view of feature importance and how values influence predictions.
   shap_values,
   X_test_scaled,
    feature_names=X.columns
) 
            #SHAP FEATURE IMPORTANCE 

import numpy as np

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
print(shap_importance)

# RANDOM FOREST FEATURE IMPORTANCE

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance")
print(rf_importance)
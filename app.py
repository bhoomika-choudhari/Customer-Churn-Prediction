# ================= IMPORTS ====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import xgboost as xgb

# ================= PAGE SETUP =================
st.set_page_config(page_title="Customer Churn Prediction 🚀", layout="wide")


# ================= LOAD DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Data Checks
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    return df


df = load_data()

# ================== EDA =======================
st.title("Customer Churn Prediction 🚀🔮")
st.header("1. Exploratory Data Analysis 📈")

# ======================= Churn Distribution =======================
st.subheader("Churn Distribution 📊")
col1, col2, col3 = st.columns([1, 4, 1])  # Margins on left and right
with col2:
    fig1, ax1 = plt.subplots(figsize=(4, 3))  # Adjust the figure size
    sns.countplot(data=df, x='Churn', palette='cool')
    ax1.set_title('Churn Count', fontsize=10)
    ax1.set_xlabel('Churn', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig1)

# ===================== Tenure vs Churn ============================
st.subheader("Tenure vs Churn ⏳")
col1, col2, col3 = st.columns([1, 4, 1])  # Margins on left and right
with col2:
    fig2, ax2 = plt.subplots(figsize=(4,3))  # Adjust the figure size
    sns.boxplot(x='Churn', y='tenure', data=df, palette='PiYG')
    ax2.set_title('Tenure by Churn', fontsize=10)
    ax2.set_xlabel('Churn', fontsize=9)
    ax2.set_ylabel('Tenure', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

# ===================== Correlation Heatmap =======================
st.subheader("Correlation Heatmap 🔥")
col1, col2, col3 = st.columns([1, 4, 1])  # Margins on left and right
with col2:
    fig3, ax3 = plt.subplots(figsize=(5,4))  # Adjust the figure size
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
                annot=True, cmap='coolwarm', fmt=".2f", ax=ax3, annot_kws={"size":6})
    ax3.set_title('Feature Correlation', fontsize= 8)
    plt.tight_layout()
    st.pyplot(fig3)

# =================  PREPROCESSING ====================
st.header("2. Data Preprocessing 📑")

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in binary_cols]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

st.success("Data Preprocessing Done ✅")

# ================== MODEL SELECTION ======================
st.header("3. Model Training & Evaluation 📡")

model_choice = st.selectbox("Choose Classifier", ["Random Forest", "XGBoost"])
model_path = f"model_{model_choice.replace(' ', '_')}.joblib"

# Train or Load Model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                  random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Hyperparameter Tuning
    if st.checkbox('Enable Hyperparameter Tuning'):
        if model_choice == "Random Forest":
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
            with st.spinner('Tuning Random Forest...'):
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f'Best Params: {grid_search.best_params_}')
        elif model_choice == "XGBoost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1]
            }
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
            with st.spinner('Tuning XGBoost...'):
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f'Best Params: {grid_search.best_params_}')
    else:
        with st.spinner('Training the model... Please wait 🧠✨'):
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            st.success('Model Trained and Saved Successfully! ✅')

# Predict
y_pred = model.predict(X_test)

# Evaluation Metrics
st.subheader("Classification Report 📋")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix 💥")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    ax_cm.set_title('Confusion Matrix', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_cm)

st.subheader("ROC-AUC Score 🏆")
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
st.write(f"ROC-AUC Score: {roc_auc:.4f}")

# ============ FEATURE IMPORTANCE ===============
st.subheader("Feature Importance 🔍")
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='cool', ax=ax_imp)
    ax_imp.set_title("Top 10 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig_imp)

# ================== CROSS VALIDATION PLOT ======================
st.header("4. Cross Validation Results 🔍")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
with st.spinner('Performing Cross-Validation... Please wait 🎯'):
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

st.write(f"Cross-Validation AUC Scores: {cv_scores}")
st.write(f"Mean AUC Score: {cv_scores.mean():.4f}")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    fig_cv, ax_cv = plt.subplots(figsize=(5, 4))
    ax_cv.plot(range(1, len(cv_scores) + 1), cv_scores, marker='*', linestyle='--', color='purple')
    ax_cv.set_xlabel('Fold', fontsize=8)
    ax_cv.set_ylabel('AUC Score', fontsize=8)
    ax_cv.set_title('Cross Validation AUC per Fold', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_cv)


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error,root_mean_squared_error
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
import scipy.stats as stats

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")
    df = df.rename(columns={
        'Extracurricular Activities': 'Activity',
        'Sample Question Papers Practiced':'Practice_Paper',
        'Performance Index' : 'Final_Score'
    })
    return df

df = load_data()

# ------------------------------
# Sidebar menu
# ------------------------------
task = st.sidebar.radio("Select Module", (
    "EDA", 
    "Data Preprocessing", 
    "Training / Optimization", 
    "Model Evaluation", 
    "Model Interpretability"
))

# ------------------------------
# Common preprocessing
# ------------------------------
feature_cols = df.select_dtypes(include=['int64','float64']).columns.drop(['Final_Score']).tolist()
X_raw = df[feature_cols].copy()
y_raw = df['Final_Score'].copy()

# Encode categorical variable
le = LabelEncoder()
df['Activity'] = le.fit_transform(df['Activity'])

# Scale numeric features
scaler = StandardScaler()
X_scaled = df[feature_cols].copy()
X_scaled[feature_cols] = scaler.fit_transform(X_scaled[feature_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Final_Score'], test_size=0.2, random_state=42)

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# ------------------------------
# Module: EDA
# ------------------------------
if task == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Numeric Feature Distribution")
    for col in feature_cols + ['Final_Score']:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"{col} Distribution")
        st.pyplot(fig)

    st.subheader("Categorical Feature Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Activity', data=df, ax=ax)
    ax.set_title("Activity Count")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Pair Plot")
    st.write("Pair plots may be slow on large datasets")
    sns.set(style="ticks")
    fig = sns.pairplot(df, vars=feature_cols + ['Final_Score'], hue='Activity', diag_kind='kde')
    st.pyplot(fig)

    st.subheader("Outlier Detection")
    for col in feature_cols + ['Final_Score']:
        fig, ax = plt.subplots()
        sns.boxplot(x='Activity', y=col, data=df, palette='Set3', ax=ax)
        ax.set_title(f"Boxplot of {col} by Activity")
        st.pyplot(fig)

    st.subheader("Missing Values Visualization")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap")
    st.pyplot(fig)

# ------------------------------
# Module: Data Preprocessing
# ------------------------------
elif task == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.subheader("Encoding & Scaling")
    st.write("Numeric features scaled, categorical encoded (Activity)")

    fig, ax = plt.subplots()
    sns.countplot(x='Activity', data=df, ax=ax)
    ax.set_title("Encoded Activity (0=No, 1=Yes)")
    st.pyplot(fig)

    st.subheader("Before vs After Scaling")
    for col in feature_cols:
        fig, ax = plt.subplots()
        sns.kdeplot(df[col], label='Original', color='blue', ax=ax)
        sns.kdeplot(X_scaled[col], label='Scaled', color='red', ax=ax)
        ax.set_title(f"{col}: Original vs Scaled")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Correlation with Final Score")
    corr_target = df_scaled = df.copy()
    corr_scaled = X_scaled.copy()
    corr_target = pd.concat([X_scaled, df['Final_Score']], axis=1).corr()['Final_Score'].sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=np.abs(corr_target[:-1]), y=corr_target.index[:-1], palette='viridis', ax=ax)
    ax.set_title("Absolute Correlation of Features with Final Score")
    ax.set_xlabel("Absolute Correlation")
    st.pyplot(fig)

    st.subheader("PCA Projection (2 Components)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Activity'], ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA Projection of Features")
    st.pyplot(fig)

# ------------------------------
# Module: Training / Optimization
# ------------------------------
elif task == "Training / Optimization":
    st.header("Training & Hyperparameter Optimization")

    st.subheader("Model Training (Default)")
    st.write("Linear Regression & Random Forest trained with default parameters")
    rmse_lr = root_mean_squared_error(y_test, y_pred_lr, )
    rmse_rf = root_mean_squared_error(y_test, y_pred_rf, )
    fig, ax = plt.subplots()
    ax.bar(['Linear Regression','Random Forest'], [rmse_lr, rmse_rf], color=['skyblue','orange'])
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Comparison")
    st.pyplot(fig)

    st.subheader("Hyperparameter Tuning for Random Forest")
    param_grid = {
        'n_estimators':[100,200,300],
        'max_depth':[None,5,10],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[1,2,4]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    y_pred_best_rf = best_rf.predict(X_test)
    rmse_best_rf = root_mean_squared_error(y_test, y_pred_best_rf, )
    st.write(f"Best Random Forest params: {grid.best_params_}")
    st.write(f"Optimized RF RMSE: {rmse_best_rf:.3f}")

# ------------------------------
# Module: Model Evaluation
# ------------------------------
elif task == "Model Evaluation":
    st.header("Model Evaluation")

    st.subheader("Prediction vs Actual")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.5, label="LR", ax=ax)
    sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, label="RF", ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Final Score")
    ax.set_ylabel("Predicted Final Score")
    ax.set_title("Prediction vs Actual")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Residuals Distribution")
    fig, ax = plt.subplots()
    sns.histplot(y_test - y_pred_lr, kde=True, color='skyblue', label='LR', ax=ax)
    sns.histplot(y_test - y_pred_rf, kde=True, color='orange', label='RF', ax=ax)
    ax.set_title("Residuals")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Q-Q Plot of Residuals")
    fig, ax = plt.subplots()
    stats.probplot(y_test - y_pred_lr, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot - Linear Regression")
    st.pyplot(fig)
    fig, ax = plt.subplots()
    stats.probplot(y_test - y_pred_rf, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot - Random Forest")
    st.pyplot(fig)

# ------------------------------
# Module: Model Interpretability
# ------------------------------
elif task == "Model Interpretability":
    st.header("Model Interpretability")

    st.subheader("Random Forest Feature Importance")
    importances = rf.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_cols, ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

    st.subheader("Linear Regression Coefficients")
    coef = lr.coef_
    fig, ax = plt.subplots()
    sns.barplot(x=coef, y=feature_cols, ax=ax)
    ax.set_title("Linear Regression Coefficients")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plots - Random Forest")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    st.write("Feature importance (bar plot)")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
    st.pyplot(bbox_inches='tight')

    st.subheader("Partial Dependence Plots - Random Forest")
    fig, ax = plt.subplots(figsize=(8,6))
    PartialDependenceDisplay.from_estimator(rf, X_train, feature_cols, ax=ax)
    st.pyplot(fig)

    st.subheader("LIME Explanation Example")
    lime_exp = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                    mode='regression')
    i = 10
    exp = lime_exp.explain_instance(X_test.values[i], rf.predict, num_features=5)
    st.write("Top 5 feature contributions for one sample:")
    st.write(exp.as_list())

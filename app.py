# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
import shap
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance Dashboard")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data(file_path="Student_Performance.csv"):
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'Extracurricular Activities': 'Activity',
        'Sample Question Papers Practiced':'Practice_Paper',
        'Performance Index' : 'Final_Score'
    })
    return df

df = load_data("Student_Performance.csv")


# -----------------------------
# Sidebar menu
# -----------------------------
task = st.sidebar.radio("Select Module", (
    "EDA", 
    "Data Preprocessing", 
    "Training / Optimization", 
    "Model Evaluation", 
    "Model Interpretability"
))

# -----------------------------
# Common preprocessing
# -----------------------------
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

# -----------------------------
# Cache and train models
# -----------------------------
@st.cache_resource
def train_models(X_train, y_train):
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    return lr, rf, gbr

lr, rf, gbr = train_models(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gbr = gbr.predict(X_test)

# -----------------------------
# Module: EDA
# -----------------------------
if task == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Numeric Feature Distributions")
    for col in feature_cols + ['Final_Score']:
        fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Feature Count")
    activity_counts = df['Activity'].value_counts().reset_index()
    activity_counts.columns = ['Activity', 'Count']
    fig = px.bar(activity_counts, x='Activity', y='Count',
                labels={'Activity':'Activity Type', 'Count':'Number of Students'},
                title="Activity Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pair Plot (Interactive Subset)")
    subset = df.sample(min(500, len(df)))
    fig = px.scatter_matrix(subset, dimensions=feature_cols + ['Final_Score'],
                            color='Activity', title="Pair Plot (Sample 500 rows)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection")
    col_select = st.selectbox("Select feature for boxplot", feature_cols + ['Final_Score'])
    fig = px.box(df, x='Activity', y=col_select, color='Activity', title=f"Boxplot of {col_select}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missing Values Overview")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing (%)": (df.isnull().mean() * 100).round(2).values,
        "Data Type": df.dtypes.values
    })
    st.dataframe(missing_df, use_container_width=True)
    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing Count",
        text="Missing (%)",
        title="Missing Values per Column",
        color="Missing Count",
        color_continuous_scale="blues"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Module: Data Preprocessing
# -----------------------------
elif task == "Data Preprocessing":
    st.header("Data Preprocessing")
    fig = px.histogram(df, x='Activity', color='Activity', title="Encoded Activity")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Before vs After Scaling")
    feature_select = st.selectbox("Select feature to visualize scaling", feature_cols)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[feature_select], name="Original", opacity=0.6))
    fig.add_trace(go.Histogram(x=X_scaled[feature_select], name="Scaled", opacity=0.6))
    fig.update_layout(barmode='overlay', title=f"{feature_select}: Original vs Scaled")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation with Target")
    corr_target = pd.concat([X_scaled, df['Final_Score']], axis=1).corr()['Final_Score'].sort_values(ascending=False)
    fig = px.bar(corr_target[1:], x=corr_target[1:].index, y=np.abs(corr_target[1:]), title="Absolute Correlation with Final Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Projection")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=df['Activity'],
                     labels={'x':'PC1','y':'PC2'}, title="PCA 2 Components")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Module: Training / Optimization
# -----------------------------
elif task == "Training / Optimization":
    st.header("Training & Optimization")
    st.write("All 3 models trained with default hyperparameters.")

    rmse_lr = np.sqrt(((y_test - y_pred_lr)**2).mean())
    rmse_rf = np.sqrt(((y_test - y_pred_rf)**2).mean())
    rmse_gbr = np.sqrt(((y_test - y_pred_gbr)**2).mean())

    st.subheader("RMSE Comparison")
    fig = px.bar(
        x=['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        y=[rmse_lr, rmse_rf, rmse_gbr],
        title="RMSE Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    # -----------------------------
    # Learning Curves Visualization
    # -----------------------------
    st.subheader("Learning Curves (Train vs Validation RMSE)")

    from sklearn.model_selection import learning_curve

    model_choice = st.selectbox(
        "Select model to view learning curve",
        ["Linear Regression", "Random Forest", "Gradient Boosting"]
    )

    if model_choice == "Linear Regression":
        model = lr
    elif model_choice == "Random Forest":
        model = rf
    else:
        model = gbr

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_scaled,
        df['Final_Score'],
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    train_rmse = -np.mean(train_scores, axis=1)
    test_rmse = -np.mean(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_rmse, mode="lines+markers",
        name="Training RMSE", line=dict(color="#1f77b4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_rmse, mode="lines+markers",
        name="Validation RMSE", line=dict(color="#ff7f0e", width=2)
    ))
    fig.update_layout(
        title=f"Learning Curve - {model_choice}",
        xaxis_title="Training Set Size",
        yaxis_title="RMSE",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Module: Model Evaluation
# -----------------------------
elif task == "Model Evaluation":
    st.header("Model Evaluation")

    st.subheader("Prediction vs Actual")
    for model_name, y_pred in zip(["Linear Regression","Random Forest","Gradient Boosting"],
                                  [y_pred_lr, y_pred_rf, y_pred_gbr]):
        fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, opacity=0.7, title=f"{model_name} Predictions")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residuals Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_test - y_pred_lr, name="Linear Regression", opacity=0.7))
    fig.add_trace(go.Histogram(x=y_test - y_pred_rf, name="Random Forest", opacity=0.7))
    fig.add_trace(go.Histogram(x=y_test - y_pred_gbr, name="Gradient Boosting", opacity=0.7))
    fig.update_layout(barmode='overlay', title="Residuals Distribution", xaxis_title="Residual")
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Absolute Error Comparison")
    abs_errors_df = pd.DataFrame({
        "Model": ["Linear Regression"]*len(y_test) + ["Random Forest"]*len(y_test) + ["Gradient Boosting"]*len(y_test),
        "Absolute Error": np.abs(np.concatenate([y_test - y_pred_lr, y_test - y_pred_rf, y_test - y_pred_gbr]))
    })
    fig = px.violin(abs_errors_df, x="Model", y="Absolute Error", box=True, points="all",
                    color="Model", title="Distribution of Absolute Errors")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance Summary")
    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
        "R²": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_gbr)],
        "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_gbr)],
        "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                 np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                 np.sqrt(mean_squared_error(y_test, y_pred_gbr))]
    })
    st.dataframe(
        metrics_df.style.format({"R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}).background_gradient(subset=["R²"], cmap="Greens"),
        use_container_width=True
    )
    fig = px.bar(metrics_df.melt(id_vars="Model"), x="variable", y="value", color="Model",
                 barmode="group", title="Model Metrics Comparison")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Module: Model Interpretability
# -----------------------------
elif task == "Model Interpretability":
    st.header("Model Interpretability")

    st.subheader("Feature Importance (Random Forest)")
    fig = px.bar(x=feature_cols, y=rf.feature_importances_, title="Random Forest Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (Gradient Boosting)")
    fig = px.bar(x=feature_cols, y=gbr.feature_importances_, title="Gradient Boosting Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Linear Regression Coefficients")
    fig = px.bar(x=feature_cols, y=lr.coef_, title="Linear Regression Coefficients")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interactive SHAP Feature Importance")
    if st.button("Compute Interactive SHAP"):
        with st.spinner("Computing SHAP values"):
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test[:300])
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({"Feature": feature_cols, "Mean |SHAP|": shap_importance}).sort_values("Mean |SHAP|", ascending=True)
            fig_bar = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation='h', color="Mean |SHAP|",
                             color_continuous_scale="viridis", title="Mean Absolute SHAP Values (Feature Importance)")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.subheader("SHAP Beeswarm Plot (Top 300 samples)")
            shap.initjs()
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_test[:300], feature_names=feature_cols, plot_type="dot", show=False)
            st.pyplot(fig)

    st.subheader("Partial Dependence Plots (RF)")
    pd_feature = st.selectbox("Select feature for PDP", feature_cols)
    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(rf, X_train, [pd_feature], ax=ax)
    st.pyplot(fig)

# -----------------------------
# LIME Explanation Section
# -----------------------------
    st.subheader("LIME Explanation Example")

    model_choice = st.selectbox(
        "Select model for LIME explanation",
        ["Linear Regression", "Random Forest", "Gradient Boosting"]
    )

    i = st.slider("Choose test sample index", 0, len(X_test)-1, 10)

    if st.button("Run LIME Explanation"):
        with st.spinner(f"Running LIME explanation for {model_choice}..."):
            lime_exp = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns,
                mode='regression'
            )

            if model_choice == "Linear Regression":
                model = lr
            elif model_choice == "Random Forest":
                model = rf
            else:
                model = gbr

            exp = lime_exp.explain_instance(
                X_test.values[i],
                model.predict,
                num_features=5
            )

            st.markdown("### Feature Contributions")
            st.write(exp.as_list())

            st.markdown("### LIME Visualization")
            fig = exp.as_pyplot_figure()
            for bar in fig.axes[0].patches:
                color = bar.get_facecolor()
                if color[0] > color[2]:
                    bar.set_facecolor((0.3, 0.6, 0.9))
                else:
                    bar.set_facecolor((0.8, 0.3, 0.3))
            st.pyplot(fig, use_container_width=True)


            st.markdown("### Interactive LIME HTML View")
            html = exp.as_html()
            st.components.v1.html(html, height=800, scrolling=True)



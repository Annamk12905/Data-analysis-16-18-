import pandas as pd
import numpy as np

# =========================
# 1. ĐỌC DỮ LIỆU
# =========================
df = pd.read_csv("mentalhealth_productivity_dataset.csv")

print("Columns:")
print(df.columns.tolist())

# =========================
# 2. THIẾU DỮ LIỆU
# =========================
missing = df.isnull().sum().reset_index()
missing.columns = ["column", "missing_values"]

missing.to_csv(
    "01_missing_values.csv",
    index=False
)

# =========================
# 3. THỐNG KÊ MÔ TẢ
# =========================
desc = pd.DataFrame({
    "mean": df.mean(numeric_only=True),
    "median": df.median(numeric_only=True),
    "std": df.std(numeric_only=True),
    "min": df.min(numeric_only=True),
    "max": df.max(numeric_only=True)
})

desc = desc.round(2)

desc.to_csv(
    "02_descriptive_statistics.csv"
)

# =========================
# 4. CORRELATION MATRIX
# =========================
corr = df.corr(numeric_only=True).round(2)

corr.to_csv(
    "03_correlation_matrix.csv"
)

# =========================
# 5. FEATURE ENGINEERING
# =========================
df["sleep_group"] = np.where(
    df["Hours_of_Sleep"] < 6,
    "Low",
    np.where(
        df["Hours_of_Sleep"] <= 8,
        "Normal",
        "High"
    )
)

df.to_csv(
    "04_cleaned_dataset.csv",
    index=False
)

# =========================
# 6. PRODUCTIVITY BY SLEEP GROUP
# =========================
group_analysis = (
    df.groupby("sleep_group")["Work_Productivity"]
    .mean()
    .reset_index()
)

group_analysis.columns = [
    "sleep_group",
    "avg_productivity"
]

group_analysis = group_analysis.round(2)

group_analysis.to_csv(
    "05_sleep_group_productivity.csv",
    index=False
)

# =========================
# 7. FEATURE IMPACT
# =========================
features = [
    "Age",
    "Hours_of_Sleep",
    "Stress_Level",
    "Exercise_Frequency"
]

impact = []

for col in features:
    corr_value = df[col].corr(
        df["Work_Productivity"]
    )

    impact.append([
        col,
        round(corr_value, 2)
    ])

impact_df = pd.DataFrame(
    impact,
    columns=[
        "feature",
        "correlation_with_productivity"
    ]
)

impact_df = impact_df.sort_values(
    by="correlation_with_productivity",
    ascending=False
)

impact_df.to_csv(
    "06_feature_impact.csv",
    index=False
)

# =========================
# 8. LINEAR REGRESSION
# =========================
X = df[features].values
y = df["Work_Productivity"].values

X_b = np.c_[
    np.ones(X.shape[0]),
    X
]

beta = (
    np.linalg.pinv(X_b.T @ X_b)
    @ X_b.T
    @ y
)

y_pred = X_b @ beta

result = df[features].copy()

result["actual_productivity"] = np.round(y, 2)
result["predicted_productivity"] = np.round(y_pred, 2)

result.to_csv(
    "07_predictions.csv",
    index=False
)

# =========================
# 9. MODEL METRICS
# =========================
mse = np.mean(
    (y - y_pred) ** 2
)

rmse = np.sqrt(mse)

metrics = pd.DataFrame({
    "MSE": [round(mse, 2)],
    "RMSE": [round(rmse, 2)]
})

metrics.to_csv(
    "08_model_metrics.csv",
    index=False
)

# =========================
# 10. COEFFICIENTS
# =========================
coef_df = pd.DataFrame({
    "feature": [
        "Intercept"
    ] + features,
    "coefficient": np.round(beta, 2)
})

coef_df.to_csv(
    "09_model_coefficients.csv",
    index=False
)

# =========================
# 11. KẾT QUẢ
# =========================
print("\nDone!")

print("\nModel Metrics")
print(metrics)

print("\nFeature Impact")
print(impact_df)

print("\nFiles Generated:")
print("01_missing_values.csv")
print("02_descriptive_statistics.csv")
print("03_correlation_matrix.csv")
print("04_cleaned_dataset.csv")
print("05_sleep_group_productivity.csv")
print("06_feature_impact.csv")
print("07_predictions.csv")
print("08_model_metrics.csv")
print("09_model_coefficients.csv")
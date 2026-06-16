import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# =========================
# 2. Clean data
# =========================
df = df.dropna()

# chuẩn hoá chữ (tránh lỗi growing/Growing)
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

# =========================
# 3. Encode categorical data
# =========================
df["Remote_Work_Score"] = df["Remote_Work_Possibility"].map({
    "no": 0,
    "hybrid": 0.5,
    "yes": 1
})

df["Hiring_Trend_Score"] = df["Hiring_Trend_2026"].map({
    "declining": 0,
    "stable": 1,
    "growing": 2
})

df["Upskilling_Needed_Score"] = df["Upskilling_Needed"].map({
    "no": 0,
    "yes": 1
})

# =========================
# 4. Convert numeric columns
# =========================
num_cols = [
    "Years_Experience",
    "AI_Replacement_Risk",
    "Future_Demand_Score",
    "Average_Salary_Usd",
    "Automation_Level",
    "Job_Growth_2030",
    "Work_Hours_Per_Week",
    "AI_Tool_Usage",
    "Performance_Score",
    "Job_Satisfaction"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 5. Basic statistics
# =========================
print("\n===== BASIC STATS =====")
print(df.describe())

# =========================
# 6. AI risk by industry
# =========================
print("\n===== AI RISK BY INDUSTRY =====")
print(
    df.groupby("Industry")["AI_Replacement_Risk"]
    .mean()
    .sort_values(ascending=False)
)

# =========================
# 7. Correlations
# =========================
print("\n===== CORRELATIONS =====")
print("AI vs Salary:",
      df["AI_Replacement_Risk"].corr(df["Average_Salary_USD"]))

print("Experience vs AI:",
      df["Years_Experience"].corr(df["AI_Replacement_Risk"]))

# =========================
# 8. Education analysis
# =========================
print("\n===== EDUCATION IMPACT =====")
print(
    df.groupby("Education_Level")["AI_Replacement_Risk"]
    .mean()
)

# =========================
# 9. Future demand
# =========================
print("\n===== TOP FUTURE JOBS =====")
print(
    df.groupby("Job_Title")["Future_Demand_Score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# =========================
# 10. Remote work effect
# =========================
print("\n===== REMOTE WORK EFFECT =====")
print(
    df.groupby("Remote_Work_Possibility")["AI_Replacement_Risk"]
    .mean()
)

# =========================
# 11. Risk classification
# =========================
def classify(x):
    if x < 0.3:
        return "Low"
    elif x < 0.7:
        return "Medium"
    return "High"

df["AI_Risk_Level"] = df["AI_Replacement_Risk"].apply(classify)

print("\n===== RISK LEVEL =====")
print(df["AI_Risk_Level"].value_counts())

# =========================
# 12. Survival Index
# =========================
df["Survival_Index"] = (
    (1 - df["AI_Replacement_Risk"]) +
    df["Future_Demand_Score"] +
    df["Remote_Work_Score"]
)

print("\n===== TOP SAFE JOBS =====")
print(
    df.sort_values("Survival_Index", ascending=False)
    [["Job_Title", "Survival_Index"]]
    .head(10)
)

# =========================
# 13. Hiring trend analysis
# =========================
print("\n===== HIRING TREND =====")
print(
    df.groupby("Hiring_Trend_2026")["Hiring_Trend_Score"]
    .count()
)

# =========================
# 14. Upskilling analysis
# =========================
print("\n===== UPSKILLING NEED =====")
print(
    df.groupby("Upskilling_Needed")["Upskilling_Needed_Score"]
    .mean()
)

# =========================
# 15. Save result
# =========================
df.to_csv("ai_job_analysis_result.csv", index=False)

print("\nDONE: Saved ai_job_analysis_result.csv")
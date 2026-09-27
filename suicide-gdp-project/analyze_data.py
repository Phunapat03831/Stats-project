from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = Path("data/processed/analysis_complete.csv")
OUT_DIR = Path("data/processed/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

variables = [
    "GDP per capita (constant 2015 US$)",
    "Unemployment (%)",
    "Current health expenditure (% of GDP)",
    "Suicide rate",
    "Government Effectiveness (0-100)",
]

# อ่านข้อมูลที่ครบทุกตัวแปร
data = pd.read_csv(DATA_FILE)

# ตรวจว่าคอลัมน์ที่ต้องใช้มีอยู่ครบ
missing_columns = [col for col in variables if col not in data.columns]
if missing_columns:
    raise ValueError(f"ไม่พบคอลัมน์: {missing_columns}")

# สถิติเชิงพรรณนา
summary = data[variables].describe().T
summary["median"] = data[variables].median()
summary.to_csv(OUT_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")

# สหสัมพันธ์เบื้องต้น
correlations = data[variables].corr(method="pearson")
correlations.to_csv(OUT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")

# กราฟกระจาย GDP ต่อหัวกับอัตราการฆ่าตัวตาย
gdp_col = "GDP per capita (constant 2015 US$)"
suicide_col = "Suicide rate"

plot_data = data[(data[gdp_col] > 0) & (data[suicide_col] >= 0)]

plt.figure(figsize=(9, 6))
plt.scatter(
    plot_data[gdp_col],
    plot_data[suicide_col],
    alpha=0.35,
    s=18,
)
plt.xscale("log")
plt.xlabel("GDP per capita (constant 2015 US$), log scale")
plt.ylabel("Suicide mortality rate")
plt.title("GDP per capita and suicide rate, 2006–2020")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "gdp_vs_suicide.png", dpi=200)
plt.close()

# กราฟ Government Effectiveness กับอัตราการฆ่าตัวตาย
gov_col = "Government Effectiveness (0-100)"

plt.figure(figsize=(9, 6))
plt.scatter(
    data[gov_col],
    data[suicide_col],
    alpha=0.35,
    s=18,
)
plt.xlabel("Government Effectiveness score (0–100)")
plt.ylabel("Suicide mortality rate")
plt.title("Government Effectiveness and suicide rate, 2006–2020")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "government_effectiveness_vs_suicide.png", dpi=200)
plt.close()

# กราฟการว่างงานกับอัตราการฆ่าตัวตาย
unemployment_col = "Unemployment (%)"

plt.figure(figsize=(9, 6))
plt.scatter(
    data[unemployment_col],
    data[suicide_col],
    alpha=0.35,
    s=18,
)
plt.xlabel("Unemployment (%)")
plt.ylabel("Suicide mortality rate")
plt.title("Unemployment and suicide rate, 2006–2020")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "unemployment_vs_suicide.png", dpi=200)
plt.close()

print("วิเคราะห์เสร็จแล้ว")
print(f"จำนวนแถวที่ใช้: {len(data):,}")
print(f"จำนวนประเทศ: {data['Country Code'].nunique():,}")
print(f"ไฟล์ผลลัพธ์อยู่ที่: {OUT_DIR}")
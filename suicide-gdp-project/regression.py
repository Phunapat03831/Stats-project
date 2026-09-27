from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DATA_FILE = Path("data/processed/analysis_complete.csv")
OUT_DIR = Path("data/processed/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# อ่านเฉพาะประเทศ-ปีที่ข้อมูลครบทั้ง 5 ตัวแปร
data = pd.read_csv(DATA_FILE)

# ตั้งชื่อคอลัมน์ให้ง่ายต่อการเขียนสมการ
data = data.rename(columns={
    "Country Code": "country_code",
    "Year": "year",
    "GDP per capita (constant 2015 US$)": "gdp_per_capita",
    "Unemployment (%)": "unemployment",
    "Current health expenditure (% of GDP)": "health_expenditure",
    "Suicide rate": "suicide_rate",
    "Government Effectiveness (0-100)": "government_effectiveness",
})

# ตัด GDP ที่ไม่เป็นบวกก่อนคำนวณ log
data = data[data["gdp_per_capita"] > 0].copy()
data["log_gdp_per_capita"] = np.log(data["gdp_per_capita"])


# โมเดล 1: ตัวแปรเศรษฐกิจ + ควบคุมประเทศและปี
formula_1 = (
    "suicide_rate ~ log_gdp_per_capita + unemployment "
    "+ health_expenditure + C(country_code) + C(year)"
)

model_1 = smf.ols(formula_1, data=data).fit(
    cov_type="cluster",
    cov_kwds={"groups": data["country_code"]},
)


# โมเดล 2: เพิ่ม Government Effectiveness
formula_2 = (
    "suicide_rate ~ log_gdp_per_capita + unemployment "
    "+ health_expenditure + government_effectiveness "
    "+ C(country_code) + C(year)"
)

model_2 = smf.ols(formula_2, data=data).fit(
    cov_type="cluster",
    cov_kwds={"groups": data["country_code"]},
)


# เก็บเฉพาะผลของตัวแปรที่เราสนใจ ไม่รวมตัวควบคุมประเทศ/ปี
main_variables = [
    "log_gdp_per_capita",
    "unemployment",
    "health_expenditure",
    "government_effectiveness",
]

rows = []

for model_name, model in [
    ("Model 1: economic variables", model_1),
    ("Model 2: plus government effectiveness", model_2),
]:
    confidence_intervals = model.conf_int()

    for variable in main_variables:
        if variable not in model.params.index:
            continue

        rows.append({
            "model": model_name,
            "variable": variable,
            "coefficient": model.params[variable],
            "clustered_standard_error": model.bse[variable],
            "p_value": model.pvalues[variable],
            "ci_95_low": confidence_intervals.loc[variable, 0],
            "ci_95_high": confidence_intervals.loc[variable, 1],
            "observations": int(model.nobs),
            "countries": data["country_code"].nunique(),
            "r_squared": model.rsquared,
        })

results = pd.DataFrame(rows)
results.to_csv(
    OUT_DIR / "regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig",
)

# เก็บรายงานฉบับเต็มไว้ตรวจสอบ
with open(OUT_DIR / "regression_model_summaries.txt", "w", encoding="utf-8") as file:
    file.write("MODEL 1: Economic variables\n")
    file.write(model_1.summary().as_text())
    file.write("\n\nMODEL 2: Plus Government Effectiveness\n")
    file.write(model_2.summary().as_text())

print("Regression เสร็จแล้ว")
print(f"จำนวนแถวที่ใช้: {int(model_2.nobs):,}")
print(f"จำนวนประเทศ: {data['country_code'].nunique():,}")
print(f"ผลตัวแปรหลัก: {OUT_DIR / 'regression_coefficients.csv'}")
print(f"รายงานโมเดลเต็ม: {OUT_DIR / 'regression_model_summaries.txt'}")
print("\nผลของตัวแปรหลัก:")
print(results.to_string(index=False))
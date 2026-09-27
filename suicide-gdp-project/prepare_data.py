from pathlib import Path
import html
import re

import pandas as pd


# ไฟล์ต้นฉบับ
RAW_DIR = Path("data/raw")
WDI_FILE = RAW_DIR / "wdi.csv"
SUICIDE_FILE = RAW_DIR / "suicide.csv"
GE_FILE = RAW_DIR / "government_effectiveness.csv"

# โฟลเดอร์และไฟล์ผลลัพธ์
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "analysis_dataset.csv"
COMPLETE_FILE = OUT_DIR / "analysis_complete.csv"
MISSING_FILE = OUT_DIR / "missing_values.csv"
COUNTRY_FILE = OUT_DIR / "country_list_to_review.csv"

START_YEAR = 2006
END_YEAR = 2020


# รหัสกลุ่มภูมิภาค รายได้ และยอดรวมจาก World Bank
AGGREGATE_CODES = {
    "AFE", "AFW", "ARB", "CSS", "CEB", "EAS", "TEA", "EAP",
    "EMU", "ECS", "TEC", "ECA", "EUU", "HPC", "HIC", "IBD",
    "IBT", "IDB", "IDX", "IDA", "LCN", "LAC", "TLA", "LDC",
    "LMY", "LIC", "LMC", "MEA", "TMN", "MNA", "MIC", "OED",
    "OSS", "PSS", "SST", "SAS", "TSA", "SSF", "TSS", "SSA",
    "UMC", "WLD",
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """แก้ชื่อคอลัมน์ที่มีรหัส HTML หรือเครื่องหมายดอกจัน"""
    df.columns = [
        re.sub(r"\*+", "", html.unescape(str(col))).strip()
        for col in df.columns
    ]
    return df


def read_csv(path: Path) -> pd.DataFrame:
    """อ่าน CSV โดยลอง encoding ที่พบบ่อย"""
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    last_error = None

    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                na_values=["..", ""],
            )
            print(f"อ่าน {path.name} ด้วย encoding: {encoding}")
            return clean_columns(df)
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def read_world_bank_api_csv(path: Path) -> pd.DataFrame:
    """อ่าน CSV จาก World Bank API ที่อาจมี metadata ก่อนหัวตาราง"""
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    expected_columns = {
        "Country Name",
        "Country Code",
        "Indicator Name",
        "Indicator Code",
    }

    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        for skip_rows in range(10):
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    skiprows=skip_rows,
                    na_values=["..", ""],
                    engine="python",
                )
                df = clean_columns(df)

                if expected_columns.issubset(set(df.columns)):
                    print(
                        f"อ่าน {path.name} ด้วย encoding {encoding} "
                        f"(ข้าม {skip_rows} บรรทัดแรก)"
                    )
                    return df

            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

    raise ValueError(
        f"หาแถวหัวตารางในไฟล์ World Bank API ไม่เจอ: {path.name}"
    )


def get_year_columns(df: pd.DataFrame) -> list[str]:
    """หาคอลัมน์ปีที่อยู่ในช่วงที่กำหนด"""
    years = []

    for col in df.columns:
        match = re.search(r"\b((?:19|20)\d{2})\b", str(col))
        if match and START_YEAR <= int(match.group(1)) <= END_YEAR:
            years.append(col)

    if not years:
        raise ValueError(
            f"ไม่พบคอลัมน์ปีช่วง {START_YEAR}-{END_YEAR}"
        )

    return years


def convert_wide_to_long(
    df: pd.DataFrame,
    country_code_column: str,
    country_name_column: str,
    value_column: str,
) -> pd.DataFrame:
    """แปลงปีที่เป็นหลายคอลัมน์ให้เป็นแถวประเทศ-ปี"""
    required = [country_code_column, country_name_column]
    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"ไม่พบคอลัมน์ที่ต้องใช้: {missing_columns}"
        )

    years = get_year_columns(df)

    long_df = df[required + years].melt(
        id_vars=required,
        value_vars=years,
        var_name="Year label",
        value_name=value_column,
    )

    long_df["Year"] = (
        long_df["Year label"]
        .astype(str)
        .str.extract(r"\b((?:19|20)\d{2})\b")[0]
        .astype(int)
    )

    long_df = long_df.rename(columns={
        country_code_column: "Country Code",
        country_name_column: "Country Name",
    })

    # ทำความสะอาดรหัสประเทศและตัดแถวที่ไม่มีรหัสออก
    long_df["Country Code"] = (
        long_df["Country Code"].astype("string").str.strip()
    )

    invalid_code = (
        long_df["Country Code"].isna()
        | long_df["Country Code"]
        .str.lower()
        .isin(["", "nan", "none"])
    )
    long_df = long_df.loc[~invalid_code].copy()

    long_df[value_column] = pd.to_numeric(
        long_df[value_column],
        errors="coerce",
    )

    return long_df[
        ["Country Code", "Country Name", "Year", value_column]
    ]


def select_series(df: pd.DataFrame, series_name: str) -> pd.DataFrame:
    """เลือกตัวชี้วัดตามชื่อ Series Name"""
    if "Series Name" not in df.columns:
        raise ValueError("ไม่พบคอลัมน์ Series Name")

    selected = df[
        df["Series Name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(series_name.strip().lower())
    ].copy()

    if selected.empty:
        available = sorted(
            df["Series Name"].dropna().astype(str).unique()
        )
        raise ValueError(
            f"หา Series Name ไม่เจอ: {series_name}\n"
            "ตัวอย่างชื่อที่พบในไฟล์:\n"
            + "\n".join(available[:30])
        )

    return selected


def get_indicator(
    df: pd.DataFrame,
    series_name: str,
    output_name: str,
) -> pd.DataFrame:
    selected = select_series(df, series_name)

    return convert_wide_to_long(
        selected,
        country_code_column="Country Code",
        country_name_column="Country Name",
        value_column=output_name,
    )


# อ่านไฟล์ต้นฉบับ
wdi = read_csv(WDI_FILE)
suicide = read_csv(SUICIDE_FILE)
ge_raw = read_world_bank_api_csv(GE_FILE)


# เลือก GDP ต่อหัว
gdp = get_indicator(
    wdi,
    "GDP per capita (constant 2015 US$)",
    "GDP per capita (constant 2015 US$)",
)

# ใช้ unemployment แบบ national estimate ตามไฟล์ที่มี
unemployment = get_indicator(
    wdi,
    "Unemployment, total (% of total labor force) (national estimate)",
    "Unemployment (%)",
)

# รายจ่ายสุขภาพปัจจุบัน คิดเป็น % ของ GDP
health = get_indicator(
    wdi,
    "Current health expenditure (% of GDP)",
    "Current health expenditure (% of GDP)",
)

# อัตราการเสียชีวิตจากการฆ่าตัวตาย
suicide_long = convert_wide_to_long(
    suicide,
    country_code_column="Country Code",
    country_name_column="Country Name",
    value_column="Suicide rate",
)

# Government Effectiveness จากไฟล์ World Bank API
ge_raw = ge_raw.rename(columns={"Indicator Name": "Series Name"})

government_effectiveness = get_indicator(
    ge_raw,
    "Government Effectiveness - Governance score (0-100)",
    "Government Effectiveness (0-100)",
)


# รวมตัวแปรโดยใช้รหัสประเทศและปี
data = gdp.copy()

for next_df in [
    unemployment,
    health,
    suicide_long,
    government_effectiveness,
]:
    next_df = next_df.drop(
        columns=["Country Name"],
        errors="ignore",
    )

    data = data.merge(
        next_df,
        on=["Country Code", "Year"],
        how="outer",
    )


# จำกัดช่วงปีและตัดกลุ่มรวมออก
data = data[data["Year"].between(START_YEAR, END_YEAR)].copy()

data = data[
    ~data["Country Code"].isin(AGGREGATE_CODES)
].copy()

data = data.sort_values(["Country Code", "Year"])


# บันทึกรายชื่อประเทศหลังกรองกลุ่มรวม
country_list = (
    data[["Country Code", "Country Name"]]
    .drop_duplicates()
    .sort_values(["Country Name", "Country Code"])
)

country_list.to_csv(
    COUNTRY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ตัวแปรที่จะใช้วิเคราะห์
variables = [
    "GDP per capita (constant 2015 US$)",
    "Unemployment (%)",
    "Current health expenditure (% of GDP)",
    "Suicide rate",
    "Government Effectiveness (0-100)",
]


# บันทึกข้อมูลทั้งหมดหลังกรองกลุ่มรวม
data.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

# บันทึกจำนวนค่าว่างในแต่ละตัวแปร
data[variables].isna().sum().rename("Missing values").to_csv(
    MISSING_FILE,
    encoding="utf-8-sig",
)

# สร้างไฟล์เฉพาะประเทศ-ปีที่มีข้อมูลครบทั้ง 5 ตัวแปร
analysis_data = data.dropna(subset=variables).copy()

analysis_data.to_csv(
    COMPLETE_FILE,
    index=False,
    encoding="utf-8-sig",
)


# สรุปผลและตรวจประเทศ-ปีซ้ำในไฟล์ข้อมูลครบ
duplicate_count = analysis_data.duplicated(
    ["Country Code", "Year"]
).sum()

print("\nจำนวนประเทศ-ปีที่มีข้อมูลครบทุกตัวแปร:", len(analysis_data))
print("จำนวนประเทศที่มีข้อมูลครบ:",
      analysis_data["Country Code"].nunique())
print("ประเทศ-ปีซ้ำในข้อมูลครบ:", duplicate_count)

print("\nจำนวนข้อมูลที่มีในแต่ละตัวแปร หลังกรองกลุ่มรวม:")
print(data[variables].notna().sum())

print(f"\nชุดข้อมูลทั้งหมด: {OUTPUT_FILE}")
print(f"ชุดข้อมูลครบสำหรับวิเคราะห์: {COMPLETE_FILE}")
print(f"รายงาน missing values: {MISSING_FILE}")
print(f"รายชื่อประเทศหลังกรอง: {COUNTRY_FILE}")
print(f"จำนวนแถวทั้งหมดหลังกรอง: {len(data):,}")
print(f"จำนวนรหัสประเทศ/เขตเศรษฐกิจ: {data['Country Code'].nunique():,}")
print(f"ช่วงปี: {data['Year'].min()}-{data['Year'].max()}")
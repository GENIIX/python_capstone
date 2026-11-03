"""
database_manipulation.py
========================
Data loading, cleaning and manipulation module (USER-DEFINED MODULE - Syllabus Unit 2.2).

Project : Analysis of Factors Associated with Diabetes Status
Dataset : CDC BRFSS 2015 - "Diabetes Health Indicators" (Kaggle, Alex Teboul)

Syllabus topics used in this file
---------------------------------
Unit 1.2  : lists, tuples, dictionaries (label maps, valid ranges)
Unit 1.3  : if / elif / else, for loops, break / continue
Unit 1.4  : user-defined functions, return values, try-except-else-finally
Unit 2.1  : os, re modules
Unit 3.1  : NumPy (percentile, clip, where, axis-wise sum)
Unit 3.2  : Pandas (read_csv, info, describe, isna, dropna/fillna, drop_duplicates,
            astype, map, cut, groupby, agg, loc/iloc, query, sort_values, to_csv)
"""

import os
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# 1. CONSTANTS  (dictionaries / lists / tuples  - Unit 1.2)
# ----------------------------------------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
CLEAN_FILE = os.path.join(OUTPUT_FOLDER, "cleaned_diabetes_data.csv")

# Dataset files that can be used (first one found is picked)
PREFERRED_FILES = (
    "diabetesData.csv",                                      # our group's file (50/50 balanced version)
    "diabetes_012_health_indicators_BRFSS2015.csv",          # 0 / 1 / 2 (3 classes)
    "diabetes_binary_health_indicators_BRFSS2015.csv",       # 0 / 1
    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
)

# Columns that must contain only 0 / 1
BINARY_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]

# Ordinal / numeric columns with their valid (min, max) according to the code-book
VALID_RANGES = {
    "BMI": (12, 98),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

# Human readable names of the yes/no risk factors (used in charts & tables)
RISK_FACTOR_LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "Smoker": "Smoker",
    "Stroke": "Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "DiffWalk": "Difficulty Walking",
    "HvyAlcoholConsump": "Heavy Alcohol Use",
    "PhysActivity": "Physically Active",
    "Fruits": "Eats Fruit Daily",
    "Veggies": "Eats Vegetables Daily",
    "NoDocbcCost": "Could Not Afford Doctor",
    "AnyHealthcare": "Has Health Coverage",
    "CholCheck": "Cholesterol Check (5 yrs)",
}

# Code-book dictionaries -> readable labels
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
    8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}
EDUCATION_LABELS = {
    1: "No schooling", 2: "Elementary", 3: "Some high school",
    4: "High school grad", 5: "Some college", 6: "College grad",
}
INCOME_LABELS = {
    1: "<$10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k",
    5: "$25-35k", 6: "$35-50k", 7: "$50-75k", 8: ">=$75k",
}
HEALTH_LABELS = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
SEX_LABELS = {0: "Female", 1: "Male"}
STATUS_LABELS_3 = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
STATUS_LABELS_2 = {0: "No Diabetes", 1: "Diabetes"}

# Display orders (lists keep the correct order in charts)
AGE_ORDER = list(AGE_LABELS.values())
EDUCATION_ORDER = list(EDUCATION_LABELS.values())
INCOME_ORDER = list(INCOME_LABELS.values())
HEALTH_ORDER = list(HEALTH_LABELS.values())
STATUS_ORDER = ["No Diabetes", "Prediabetes", "Diabetes"]
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
RISK_LEVEL_ORDER = ["Low (0-1)", "Moderate (2-3)", "High (4+)"]

# Columns that add one point to the Risk_Factor_Count (clinical / lifestyle risk)
RISK_SCORE_COLUMNS = ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]


# ----------------------------------------------------------------------------------
# 2. LOADING THE DATASET
# ----------------------------------------------------------------------------------
def find_dataset(data_folder=DATA_FOLDER):
    """Look inside the data folder (os module) and return the path of the dataset CSV."""
    if not os.path.isdir(data_folder):
        return None

    files_found = os.listdir(data_folder)

    for name in PREFERRED_FILES:                       # exact known file names first
        if name in files_found:
            return os.path.join(data_folder, name)

    for name in files_found:                           # any other csv that looks right
        if name.lower().endswith(".csv") and "diabetes" in name.lower():
            return os.path.join(data_folder, name)
    return None


def load_data(path):
    """Read the CSV with pandas. Uses try-except-else-finally (Unit 1.4)."""
    df = None
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The file is empty: {path}")
    except Exception as error:                         # any other run-time error
        print(f"[ERROR] Could not read file: {error}")
    else:
        print(f"[OK] Loaded '{os.path.basename(path)}'  ->  {df.shape[0]:,} rows x {df.shape[1]} columns")
    finally:
        print("[INFO] Data loading step finished.")
    return df


def standardize_column_names(df):
    """Remove spaces / special characters from column names using the re module."""
    new_names = {col: re.sub(r"\W+", "_", str(col).strip()) for col in df.columns}
    return df.rename(columns=new_names)


def identify_target(df):
    """Return the name of the diabetes column present in the dataset."""
    for name in ("Diabetes_012", "Diabetes_binary", "Diabetes_Binary"):
        if name in df.columns:
            return name
    raise ValueError("No diabetes target column found (expected Diabetes_012 or Diabetes_binary).")


# ----------------------------------------------------------------------------------
# 3. INSPECTION
# ----------------------------------------------------------------------------------
def inspect_data(df):
    """Print basic information of the DataFrame (head, shape, info, describe ...)."""
    print("\n--- First 5 rows (head) ---")
    print(df.head().to_string())
    print("\n--- Last 3 rows (tail) ---")
    print(df.tail(3).to_string())
    print(f"\nShape (rows, columns) : {df.shape}")
    print(f"Number of cells       : {df.size:,}")
    print(f"Memory used           : {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print("\n--- info() ---")
    df.info()
    print("\n--- describe() ---")
    print(df.describe().T.round(2).to_string())
    print(f"\nDuplicate rows        : {df.duplicated().sum():,}")
    print(f"Total missing values  : {int(df.isna().sum().sum())}")


def missing_value_report(df):
    """Count and percentage of missing values per column (isna / sum)."""
    report = pd.DataFrame({
        "Missing": df.isna().sum(),
        "Percent": (df.isna().sum() / len(df) * 100).round(3),
    })
    return report.sort_values("Missing", ascending=False)


# ----------------------------------------------------------------------------------
# 4. CLEANING STEPS  (each step is its own small function)
# ----------------------------------------------------------------------------------
def validate_ranges(df, log):
    """Values outside the code-book range are marked as NaN (to be imputed later)."""
    ranges = dict(VALID_RANGES)
    for col in BINARY_COLUMNS:                         # binary columns must be 0 or 1
        ranges[col] = (0, 1)

    total_invalid = 0
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue                                   # skip columns that are not present
        invalid = (df[col] < low) | (df[col] > high)   # relational + logical operators
        count = int(invalid.sum())
        if count > 0:
            df.loc[invalid, col] = np.nan
            total_invalid += count
            log.append(f"  - {col}: {count} value(s) outside [{low}, {high}] set to NaN")
    log.append(f"Range validation done: {total_invalid} invalid value(s) found.")
    return df


def impute_missing(df, log):
    """Fill missing values: mode for coded/binary columns, median for numeric columns."""
    coded_columns = BINARY_COLUMNS + ["GenHlth", "Age", "Education", "Income"]
    filled = 0
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if col in coded_columns:
            fill_value = df[col].mode()[0]
            method = "mode"
        else:
            fill_value = df[col].median()
            method = "median"
        df[col] = df[col].fillna(fill_value)
        filled += n_missing
        log.append(f"  - {col}: {n_missing} missing value(s) imputed with {method} ({fill_value})")
    log.append(f"Missing value handling done: {filled} value(s) imputed.")
    return df


def remove_duplicates(df, log):
    """Drop exact duplicate respondents (all 22 answers identical)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.append(f"Duplicate rows removed: {removed:,}  ({before:,} -> {len(df):,} rows)")
    return df


def optimise_dtypes(df, target, log):
    """Convert float columns that only hold whole numbers to small integers (saves memory)."""
    before = df.memory_usage(deep=True).sum() / 1024 ** 2
    int_columns = BINARY_COLUMNS + ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income", target]
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    df["BMI"] = df["BMI"].astype("float64")
    after = df.memory_usage(deep=True).sum() / 1024 ** 2
    log.append(f"Data types optimised (float64 -> int8): {before:.1f} MB -> {after:.1f} MB")
    return df


def handle_bmi_outliers(df, log):
    """
    Detect BMI outliers with the IQR rule using NumPy.
    - BMI values are clinically plausible (validated 12-98) so the ORIGINAL column is kept.
    - 'BMI_Outlier'  : flag (1 = outlier)
    - 'BMI_Capped'   : BMI clipped to the IQR fences (used by parametric tests).
    """
    q1, q3 = np.percentile(df["BMI"], [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flag = (df["BMI"] < lower) | (df["BMI"] > upper)
    df["BMI_Outlier"] = flag.astype("int8")
    df["BMI_Capped"] = np.clip(df["BMI"], lower, upper)
    log.append(f"BMI outliers (IQR rule): Q1={q1:.1f}, Q3={q3:.1f}, fences=[{lower:.1f}, {upper:.1f}] "
               f"-> {int(flag.sum()):,} flagged ({flag.mean() * 100:.2f}%), kept + BMI_Capped created")
    return df


# ----------------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ----------------------------------------------------------------------------------
def engineer_features(df, target, log):
    """Create readable labels and new analytical columns."""
    # --- diabetes status ---------------------------------------------------------
    if target == "Diabetes_012":
        df["Diabetes_Status"] = df[target].map(STATUS_LABELS_3)
        df["Diabetes_Binary"] = np.where(df[target] >= 1, 1, 0).astype("int8")   # 1 = prediabetes or diabetes
    else:
        df["Diabetes_Binary"] = df[target].astype("int8")
        df["Diabetes_Status"] = df["Diabetes_Binary"].map(STATUS_LABELS_2)
    df = df.drop(columns=[target])                     # drop original, keep clean versions

    # --- label columns using dictionaries ---------------------------------------
    df["Sex_Label"] = df["Sex"].map(SEX_LABELS)
    df["Age_Group"] = df["Age"].map(AGE_LABELS)
    df["Education_Level"] = df["Education"].map(EDUCATION_LABELS)
    df["Income_Group"] = df["Income"].map(INCOME_LABELS)
    df["General_Health"] = df["GenHlth"].map(HEALTH_LABELS)

    # --- BMI category (WHO classes) using pd.cut ----------------------------------
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 200],
                                labels=BMI_ORDER, right=False).astype(str)

    # --- risk score: NumPy axis-wise sum + np.where --------------------------------
    risk_matrix = df[RISK_SCORE_COLUMNS].to_numpy()               # 2-D NumPy array
    base_count = risk_matrix.sum(axis=1)                          # sum across columns (axis=1)
    obese_point = np.where(df["BMI"] >= 30, 1, 0)
    inactive_point = np.where(df["PhysActivity"] == 0, 1, 0)
    df["Risk_Factor_Count"] = (base_count + obese_point + inactive_point).astype("int8")
    df["Risk_Level"] = pd.cut(df["Risk_Factor_Count"], bins=[-1, 1, 3, 20],
                              labels=RISK_LEVEL_ORDER).astype(str)

    # --- other derived columns ------------------------------------------------------
    df["Healthy_Lifestyle_Score"] = (df["PhysActivity"] + df["Fruits"] + df["Veggies"]).astype("int8")
    df["Poor_Health_Days_Total"] = (df["PhysHlth"] + df["MentHlth"]).astype("int16")

    log.append("Feature engineering done: Diabetes_Status, Diabetes_Binary, Sex_Label, Age_Group, "
               "Education_Level, Income_Group, General_Health, BMI_Category, Risk_Factor_Count, "
               "Risk_Level, Healthy_Lifestyle_Score, Poor_Health_Days_Total")
    return df


# ----------------------------------------------------------------------------------
# 6. MASTER CLEANING FUNCTION (calls the small functions above - function nesting, Unit 1.4)
# ----------------------------------------------------------------------------------
def clean_dataset(df_raw):
    """Run the full cleaning pipeline and return (clean_df, log_lines)."""
    log = []
    df = standardize_column_names(df_raw.copy())
    target = identify_target(df)
    log.append(f"Raw data: {df.shape[0]:,} rows x {df.shape[1]} columns | target column = '{target}'")

    df = validate_ranges(df, log)
    df = impute_missing(df, log)
    df = remove_duplicates(df, log)
    df = optimise_dtypes(df, target, log)
    df = handle_bmi_outliers(df, log)
    df = engineer_features(df, target, log)

    log.append(f"Final clean data: {df.shape[0]:,} rows x {df.shape[1]} columns, "
               f"missing values = {int(df.isna().sum().sum())}")
    return df, log


def save_clean_data(df, log=None, path=CLEAN_FILE):
    """Export the cleaned DataFrame (and the cleaning log) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"[SAVED] Cleaned dataset -> {path}  ({size_mb:.1f} MB)")
    if log:
        log_path = os.path.join(os.path.dirname(path), "cleaning_log.txt")
        with open(log_path, "w", encoding="utf-8") as file:
            file.write("\n".join(log))
        print(f"[SAVED] Cleaning log     -> {log_path}")


def load_clean_data(path=CLEAN_FILE):
    """Read the cleaned CSV back (returns None if it does not exist yet)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ----------------------------------------------------------------------------------
# 7. DATA MANIPULATION / SUMMARY FUNCTIONS (groupby, agg, loc, iloc, query, sort)
# ----------------------------------------------------------------------------------
def prevalence_by(df, group_col, order=None):
    """Number of cases, total people and diabetes rate (%) for every group of a column."""
    table = df.groupby(group_col)["Diabetes_Binary"].agg(Cases="sum", Total="count", Rate="mean")
    table["Rate_Percent"] = (table["Rate"] * 100).round(2)
    table = table.drop(columns="Rate")
    if order is not None:
        table = table.reindex(order)
    return table


def summary_tables(df):
    """Return a dictionary of the key group-by tables used in the report."""
    tables = {
        "By Age Group": prevalence_by(df, "Age_Group", AGE_ORDER),
        "By Sex": prevalence_by(df, "Sex_Label"),
        "By Income Group": prevalence_by(df, "Income_Group", INCOME_ORDER),
        "By Education": prevalence_by(df, "Education_Level", EDUCATION_ORDER),
        "By General Health": prevalence_by(df, "General_Health", HEALTH_ORDER),
        "By BMI Category": prevalence_by(df, "BMI_Category", BMI_ORDER),
        "By Risk Level": prevalence_by(df, "Risk_Level", RISK_LEVEL_ORDER),
    }
    return tables


def status_profile(df):
    """Mean / median / std of key numeric columns for every diabetes status (groupby + agg)."""
    columns = ["BMI", "MentHlth", "PhysHlth", "Age", "Income", "Risk_Factor_Count"]
    profile = df.groupby("Diabetes_Status")[columns].agg(["mean", "median", "std"]).round(2)
    return profile


def risk_factor_rates(df):
    """Diabetes rate (%) among people WITH and WITHOUT every yes/no factor."""
    rows = []
    for column, label in RISK_FACTOR_LABELS.items():       # loop over dictionary items
        yes_rate = df.loc[df[column] == 1, "Diabetes_Binary"].mean() * 100
        no_rate = df.loc[df[column] == 0, "Diabetes_Binary"].mean() * 100
        rows.append({"Factor": label, "Column": column,
                     "Rate_If_Yes": round(yes_rate, 2), "Rate_If_No": round(no_rate, 2),
                     "Difference": round(yes_rate - no_rate, 2)})
    return pd.DataFrame(rows).sort_values("Difference", ascending=False).reset_index(drop=True)


def high_risk_group(df, min_age_code=9, min_risk=4):
    """Filter older respondents with many risk factors (logical operators + loc)."""
    mask = (df["Age"] >= min_age_code) & (df["Risk_Factor_Count"] >= min_risk)
    return df.loc[mask, ["Age_Group", "Sex_Label", "BMI", "Risk_Factor_Count", "General_Health", "Diabetes_Status"]]


def top_bmi_respondents(df, n=5):
    """sort_values + head: the n respondents with the highest BMI."""
    return df.sort_values("BMI", ascending=False).head(n)[["Age_Group", "Sex_Label", "BMI", "Diabetes_Status"]]


def query_example(df):
    """Use query() to find women aged 60+ with high BP and high cholesterol."""
    return df.query("Sex == 0 and Age >= 9 and HighBP == 1 and HighChol == 1")


def correlation_with_target(df):
    """Correlation of every numeric column with Diabetes_Binary (sorted)."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()["Diabetes_Binary"].drop("Diabetes_Binary")
    return corr.sort_values(ascending=False).round(3)


def print_manipulation_demo(df):
    """Print several manipulation outputs (used in the app menu & viva demo)."""
    print("\n=== Diabetes rate (%) by group (groupby + agg) ===")
    for name, table in summary_tables(df).items():
        print(f"\n{name}")
        print(table.to_string())

    print("\n=== Profile of each diabetes status (groupby + agg) ===")
    print(status_profile(df).to_string())

    print("\n=== Risk factors: diabetes rate if present vs absent ===")
    print(risk_factor_rates(df).to_string(index=False))

    print("\n=== Correlation of each numeric column with Diabetes_Binary ===")
    print(correlation_with_target(df).to_string())

    high = high_risk_group(df)
    print(f"\n=== High-risk group (age >= 60 and >= 4 risk factors): {len(high):,} people ===")
    print(f"Diabetes rate in this group: {high['Diabetes_Status'].ne('No Diabetes').mean() * 100:.1f}%")
    print(high.head().to_string())

    print("\n=== Top 5 BMI values (sort_values) ===")
    print(top_bmi_respondents(df).to_string())

    print(f"\n=== query(): women 60+ with high BP and high cholesterol -> {len(query_example(df)):,} people ===")
    print("\n=== iloc: rows 0-2, first 6 columns ===")
    print(df.iloc[0:3, 0:6].to_string())

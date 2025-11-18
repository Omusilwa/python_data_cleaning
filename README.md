# Data Cleaning Process:


Goal: produce an accurate, consistent, analysis‑ready dataset for health policy, epidemiology, ML, and global health research.

---
## Setup & Reproducibility
- Ensure anyone can rerun your cleaning exactly.
``` bash
# recommended environment (example)
pip install pandas numpy missingno ydata-profiling pandera python-dateutil
```
---
``` python

import pandas as pd
import numpy as np
from datetime import datetime
import missingno as msno
from ydata_profiling import ProfileReport
import pandera as pa
from pandera import Column, Check
```
---
### 1) Understand the Data Structure
- Know what each field means before touching it.
- Actions:
  - Identify core variables: disease, patient demographics, time period, facility/location, outcomes.
  - Draft a data dictionary (names, types, units, allowed values).

Code:
``` python

df = pd.read_csv("path/to/raw.csv")

# Fast orientation
df.shape
df.info()
df.head()
df.describe(include="all")
df.columns.to_list()

# Start a simple data dictionary scaffold
data_dict = pd.DataFrame({
    "column": df.columns,
    "dtype_inferred": [str(t) for t in df.dtypes],
    "description": "",        # fill manually or from source docs
    "unit": "",               # add if applicable
    "allowed_values": ""      # for categoricals
})
```
---
### 2) Profile & Visualize Missingness
- See where and how data is missing to choose good strategies.
- Actions:
  - Quantify and visualize missingness.
  - Decide what’s critical (dates, disease, location).

Code:
```python
# Quantify
null_counts = df.isna().sum().sort_values(ascending=False)
null_pct = df.isna().mean().sort_values(ascending=False)

# Visualize (skips if no missing values)
if df.isna().values.any():
    msno.bar(df)
    msno.matrix(df)
    msno.heatmap(df)

# One‑click profiling (HTML report)
profile = ProfileReport(df, title="Dataset Profiling", explorative=True)
profile.to_file("reports/01_profiling_raw.html")
```
---
### 3) Define Validation Rules (Schema)
- Make implicit assumptions explicit (types, ranges, allowed values).
- Actions: Use a lightweight schema to catch errors early.

Code (example):
```python
schema = pa.DataFrameSchema({
    "patient_id": Column(str, nullable=False),
    "age": Column(float, Check.in_range(0, 120), nullable=True),
    "sex": Column(str, Check.isin(["female", "male", "other"]), nullable=True),
    "disease": Column(str, nullable=False),
    "date_visit": Column(str, nullable=False),  # will cast to datetime later
    "facility": Column(str, nullable=True),
    "outcome": Column(str, nullable=True),
})

# Validate current state (won’t coerce yet)
schema.validate(df, lazy=True)
```
---
### 4) Standardize Data Types & Formats
- Consistent types enable time‑series, grouping, and modeling.
- Actions:
  - Dates → YYYY-MM-DD
  - Numeric fields → numeric dtypes
  - Categorical text → consistent case/whitespace

Code:

```python
# Dates
df["date_visit"] = pd.to_datetime(df["date_visit"], errors="coerce", utc=False).dt.date

# Numerics (example placeholders)
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Text normalization
for c in ["disease", "facility", "outcome"]:
    df[c] = (df[c]
             .astype("string")
             .str.strip()
             .str.lower())  # choose a single case convention

# Optional: categorical dtype to save memory
cat_cols = ["disease", "facility", "outcome", "sex"]
for c in cat_cols:
    df[c] = df[c].astype("category")
```
---
### 4) Standardize Data Types & Formats
- Consistent types enable time‑series, grouping, and modeling.
Actions:
  - Dates → YYYY-MM-DD
  - Numeric fields → numeric dtypes
  - Categorical text → consistent case/whitespace

Code:
```python
# Dates
df["date_visit"] = pd.to_datetime(df["date_visit"], errors="coerce", utc=False).dt.date

# Numerics (example placeholders)
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Text normalization
for c in ["disease", "facility", "outcome"]:
    df[c] = (df[c]
             .astype("string")
             .str.strip()
             .str.lower())  # choose a single case convention

# Optional: categorical dtype to save memory
cat_cols = ["disease", "facility", "outcome", "sex"]
for c in cat_cols:
    df[c] = df[c].astype("category")
```
---
### 5) Handle Missing Data (Decision‑guided)
- Prevent biased estimates and model leakage.
Actions (typical rules—adapt for context):
  - Critical fields (date, disease, location): prefer impute carefully or flag/exclude, not blind fills.
  - Low‑impact fields: impute with domain‑appropriate methods.
  - Always create flags for imputation.

Code:
```python
# Example: create flags so downstream users know what was imputed
for c in ["age", "outcome"]:
    df[f"{c}_was_missing"] = df[c].isna()

# Example strategies (placeholders)
df["age"] = df["age"].fillna(df["age"].median())            # numeric
df["outcome"] = df["outcome"].fillna("unknown")             # categorical

# Critical fields — choose conservative approach (illustrative)
df = df[~df["disease"].isna()]                              # or df["disease"].fillna("unspecified") with caution
df = df[~df["date_visit"].isna()]
```
---
### 6) Correct Data Entry Errors & Inconsistent Values
- Remove impossible/implausible values; unify synonyms and typos.
Actions:
- Range checks (e.g., negative age).
- Controlled vocabularies (e.g., “malaria”, “plasmodium falciparum malaria” → “malaria”).

Code:
```python
# Basic plausibility checks (example)
df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 120))]

# Standardize common variants (placeholder mapping)
disease_map = {
    "malaria": "malaria",
    "maleria": "malaria",
    "plasmodium falciparum": "malaria",
    # ...
}
df["disease"] = df["disease"].replace(disease_map)
```
---
### 7) Outlier Detection (QA, not auto-delete)
- Identify values that may be errors or true extremes; review with domain context.

Actions: Start with IQR for numeric vitals/labs; review before dropping.

Code (example placeholder):
```python
col = "some_numeric_measure"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

# Flag outliers (do not drop by default)
df[f"{col}_is_outlier"] = ~df[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```
---
### 8) De‑duplicate Records
- Prevent overcounting patients/admissions and inflated prevalence.
Actions:
  - Define a unique key (e.g., patient_id + date_visit + disease + facility).
  - Drop duplicates, keep latest or most complete.

Code:
```python
# Count duplicates
dup_mask = df.duplicated(subset=["patient_id", "date_visit", "disease", "facility"], keep=False)

# Resolve: choose keep="first" or use completeness rules
df = df.drop_duplicates(subset=["patient_id", "date_visit", "disease", "facility"], keep="first")
```
---
### 9) Re‑validate, Re‑profile, and Log Changes
- Verify the dataset now matches expectations; create an audit trail.
Actions:
  - Re‑run schema with coercion.
  - Re‑run profiling on cleaned data.
  - Write a cleaning log (what changed, why, how many rows affected).

Code:
```python
# Tighten schema (example coercion via pandas already done)
schema.validate(df, lazy=True)

# New profile after cleaning
clean_profile = ProfileReport(df, title="Dataset Profiling (Cleaned)", explorative=True)
clean_profile.to_file("reports/02_profiling_clean.html")

# Simple change log example
log = {
    "dropped_rows_missing_critical": int((~df["disease"].notna()).sum() + (~df["date_visit"].notna()).sum()),
    "duplicates_removed": int(dup_mask.sum())
    # add more counters as you apply rules
}
```
---
### 10) Save Clean Data & Documentation
- Make the cleaned data easy to reuse and automate.
Actions:
  - Save to `data/clean/` as CSV and/or Parquet.
  - Export the final data dictionary.
  - Document assumptions and decisions.

Code:
```python
df.to_csv("data/clean/hospital_clean.csv", index=False)
df.to_parquet("data/clean/hospital_clean.parquet", index=False)

data_dict.to_csv("docs/data_dictionary.csv", index=False)
```
---
---
## Appendix: Quick Commands Reference
Inspect:
```python

df.shape; df.info(); df.head(); df.describe(include="all"); df.columns
```
Missing values:
```python
df.isna().sum(); df.dropna(subset=["critical_col"]); df["col"].fillna(df["col"].median())
```
Types & Formats:
```python
df["date_col"] = pd.to_datetime(df["date_col"], errors="coerce").dt.date
df["num_col"] = pd.to_numeric(df["num_col"], errors="coerce")
df["text_col"] = df["text_col"].str.strip().str.lower()
```
Duplicates:
```python
df.duplicated(subset=[...]).sum()
df = df.drop_duplicates(subset=[...], keep="first")
```
Outliers (IQR):
```python
Q1 = df["col"].quantile(0.25); Q3 = df["col"].quantile(0.75); IQR = Q3 - Q1
df["col_is_outlier"] = ~df["col"].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```
missingno & ydata_profiling:
```python
if df.isna().values.any():
    msno.bar(df); msno.matrix(df); msno.heatmap(df)

ProfileReport(df, title="Profiling", explorative=True).to_file("reports/profiling.html")
```

"""
Pandas Basics for Data Science
================================
Covers: Series, DataFrame creation, indexing, data cleaning, merging,
        groupby aggregations, and basic time-series handling.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Series
# ---------------------------------------------------------
print("=== Series ===")

s = pd.Series([10, 20, 30, 40, 50], index=list("abcde"))
print(s)
print("Index:", s.index.tolist())
print("Values:", s.values)
print("s['c'] =", s["c"])
print("s[s > 25]:\n", s[s > 25])

# ---------------------------------------------------------
# 2. DataFrame Creation
# ---------------------------------------------------------
print("\n=== DataFrame Creation ===")

data = {
    "name":       ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":        [25, 30, 35, 28, 22],
    "department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
    "salary":     [70000, 55000, 90000, 60000, 48000],
    "score":      [88.5, 72.0, 95.0, np.nan, 65.5],
}

df = pd.DataFrame(data)
print(df)
print("\nShape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nInfo:")
df.info()

# ---------------------------------------------------------
# 3. Indexing & Selection
# ---------------------------------------------------------
print("\n=== Indexing & Selection ===")

print("Select 'name' column:\n", df["name"].tolist())
print("\nSelect multiple columns:\n", df[["name", "salary"]])
print("\nRows where salary > 60000:\n", df[df["salary"] > 60_000])
print("\n.loc[1:3, ['name', 'department']]:\n", df.loc[1:3, ["name", "department"]])
print("\n.iloc[0:2, 0:3]:\n", df.iloc[0:2, 0:3])

# ---------------------------------------------------------
# 4. Descriptive Statistics
# ---------------------------------------------------------
print("\n=== Descriptive Statistics ===")

print(df[["age", "salary", "score"]].describe())

# ---------------------------------------------------------
# 5. Handling Missing Data
# ---------------------------------------------------------
print("\n=== Handling Missing Data ===")

print("Null counts:\n", df.isnull().sum())

df_filled = df.copy()
df_filled["score"] = df_filled["score"].fillna(df_filled["score"].mean())
print("After filling NaN with mean:\n", df_filled["score"].tolist())

df_dropped = df.dropna()
print(f"Rows after dropna: {len(df_dropped)} (was {len(df)})")

# ---------------------------------------------------------
# 6. Adding & Modifying Columns
# ---------------------------------------------------------
print("\n=== Adding & Modifying Columns ===")

df_filled["senior"] = df_filled["age"] >= 30
df_filled["salary_k"] = df_filled["salary"] / 1000
df_filled["name_upper"] = df_filled["name"].str.upper()
print(df_filled[["name_upper", "age", "salary_k", "senior"]])

# ---------------------------------------------------------
# 7. Sorting
# ---------------------------------------------------------
print("\n=== Sorting ===")

print(df_filled.sort_values("salary", ascending=False)[["name", "salary"]].to_string(index=False))

# ---------------------------------------------------------
# 8. GroupBy & Aggregation
# ---------------------------------------------------------
print("\n=== GroupBy & Aggregation ===")

dept_stats = (
    df_filled.groupby("department")["salary"]
    .agg(count="count", mean="mean", min="min", max="max")
    .round(2)
)
print(dept_stats)

pivot = df_filled.pivot_table(values="salary", index="department", aggfunc=["mean", "count"])
print("\nPivot table:\n", pivot)

# ---------------------------------------------------------
# 9. Merging DataFrames
# ---------------------------------------------------------
print("\n=== Merging DataFrames ===")

projects = pd.DataFrame({
    "name":    ["Alice", "Bob", "Alice", "Charlie"],
    "project": ["Alpha", "Beta", "Gamma", "Alpha"],
})

merged = df_filled.merge(projects, on="name", how="left")
print(merged[["name", "department", "project"]].to_string(index=False))

# ---------------------------------------------------------
# 10. Apply & Map
# ---------------------------------------------------------
print("\n=== Apply & Map ===")


def salary_band(sal):
    if sal >= 80_000:
        return "High"
    elif sal >= 60_000:
        return "Mid"
    return "Low"


df_filled["band"] = df_filled["salary"].apply(salary_band)
print(df_filled[["name", "salary", "band"]].to_string(index=False))

# ---------------------------------------------------------
# 11. Simple Time Series
# ---------------------------------------------------------
print("\n=== Time Series ===")

dates = pd.date_range(start="2024-01-01", periods=6, freq="ME")
ts = pd.Series([100, 110, 105, 120, 115, 130], index=dates)
print(ts)
print("Rolling mean (2):\n", ts.rolling(2).mean())

print("\nDone! Pandas basics covered successfully.")

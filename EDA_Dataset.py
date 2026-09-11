"""
Sleep Health and Lifestyle dataset - Week 2 data analysis.

Source: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
374 rows, 13 columns, synthetic data created by the dataset author.

Question explored: which lifestyle factors line up with sleep quality,
and can a simple regression predict it?
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CSV = "data/Sleep_health_and_lifestyle_dataset.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

os.makedirs("images", exist_ok=True)


def section(name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


# --- Load ---

section("1. LOAD")

df = pd.read_csv(CSV)
print("Rows and columns:", df.shape)


# --- Inspect ---

section("2. INSPECT")

print(df.head())
print()
df.info()
print("\nUnique values per column:")
print(df.nunique())
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nFully identical rows:", df.duplicated().sum())
print("Identical rows ignoring Person ID:", df.drop(columns="Person ID").duplicated().sum())

# FINDING 1: Sleep Disorder has 219 missing values because pandas reads
# "None" as NaN. I don't drop them because that would remove 219 of 374
# records and leave only Insomnia and Sleep Apnea cases. So we will have only people with disorders, instead I will fill the missing values with "No Disorder" to keep the full dataset for analysis.

# FINDING 2: No rows are fully identical because Person ID is unique. If I ignore Person ID, 242 rows repeat an earlier profile, leaving 132 distinct profiles
# I'll keep them for the main analysis but run the regression again without them just to test how much they affect the results.


# --- Clean ---

section("3. CLEAN")


df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No Disorder")
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")

# Blood Pressure arrives as text like "126/83", so split it into two numbers.
df[["Systolic", "Diastolic"]] = (
    df["Blood Pressure"].str.split("/", expand=True).astype(int)
)

print("BMI categories after merging:")
print(df["BMI Category"].value_counts())


# --- Explore ---

section("4. EXPLORE")

print("\nSleep disorder distribution:")
print(df["Sleep Disorder"].value_counts())
print("\nOccupation distribution:")
print(df["Occupation"].value_counts())


# --- Pandas analysis ---

section("5. PANDAS ANALYSIS")

print("\nAverage sleep quality by stress level:")
stress_summary = df.groupby("Stress Level")["Quality of Sleep"].mean()
print(stress_summary.round(2))

high_stress = df[df["Stress Level"] >= 7]
low_stress = df[df["Stress Level"] <= 4]

print("\nHigh-stress records:", len(high_stress))
print("Average sleep duration:", round(high_stress["Sleep Duration"].mean(), 2))
print("Average sleep quality:", round(high_stress["Quality of Sleep"].mean(), 2))

print("\nLow-stress records:", len(low_stress))
print("Average sleep duration:", round(low_stress["Sleep Duration"].mean(), 2))
print("Average sleep quality:", round(low_stress["Quality of Sleep"].mean(), 2))

short_sleep = df[df["Sleep Duration"] < 6]
print("\nRecords with under 6 hours of sleep:")
print(short_sleep[["Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Stress Level"]])

# record_count, not participant_count - see FINDING 2
sleep_by_bmi = df.groupby("BMI Category").agg(
    average_sleep=("Sleep Duration", "mean"),
    average_quality=("Quality of Sleep", "mean"),
    record_count=("Person ID", "count"),
)
print("\nSleep metrics by BMI category:")
print(sleep_by_bmi.round(2))

unique_profiles = df.drop(columns="Person ID").drop_duplicates()
print("\nBMI counts using distinct profiles only:")
print(unique_profiles["BMI Category"].value_counts())


# --- Polars analysis ---

section("6. POLARS ANALYSIS")

# Polars keeps "None" as text, so tell it to read that as null to match pandas.
pl_df = pl.read_csv(CSV, null_values={"Sleep Disorder": "None"})
pl_df = pl_df.with_columns(
    pl.col("Sleep Disorder").fill_null("No Disorder"),
    pl.col("BMI Category").replace("Normal Weight", "Normal"),
)

print("\nAverage sleep quality by stress level:")
print(
    pl_df.group_by("Stress Level")
    .agg(pl.col("Quality of Sleep").mean().round(2).alias("average_sleep_quality"))
    .sort("Stress Level")
)

pl_high = pl_df.filter(pl.col("Stress Level") >= 7)
pl_low = pl_df.filter(pl.col("Stress Level") <= 4)

print("\nHigh-stress records:", pl_high.height)
print("Average sleep duration:", round(pl_high["Sleep Duration"].mean(), 2))
print("Average sleep quality:", round(pl_high["Quality of Sleep"].mean(), 2))

print("\nLow-stress records:", pl_low.height)
print("Average sleep duration:", round(pl_low["Sleep Duration"].mean(), 2))
print("Average sleep quality:", round(pl_low["Quality of Sleep"].mean(), 2))

print("\nRecords with under 6 hours of sleep:")
print(
    pl_df.filter(pl.col("Sleep Duration") < 6)
    .select(["Age", "Occupation", "Sleep Duration","Quality of Sleep", "Stress Level"])
)

print("\nSleep metrics by BMI category:")
print(
    pl_df.group_by("BMI Category")
    .agg(
        pl.col("Sleep Duration").mean().round(2).alias("average_sleep"),
        pl.col("Quality of Sleep").mean().round(2).alias("average_quality"),
        pl.len().alias("record_count"),
    )
    .sort("BMI Category")
)

profile_columns = [c for c in pl_df.columns if c != "Person ID"]
print("\nBMI counts using distinct profiles only:")
print(
    pl_df.unique(subset=profile_columns)
    .group_by("BMI Category")
    .agg(pl.len().alias("record_count"))
    .sort("BMI Category")
)


# --- Performance comparison ---

section("7. PANDAS VS POLARS PERFORMANCE")

# Same filtering and grouping steps in both libraries, nothing printed,
# so the timings measure the work rather than the output.


def pandas_analysis():
    df.groupby("Stress Level")["Quality of Sleep"].mean()
    df[df["Stress Level"] >= 7]
    df[df["Stress Level"] <= 4]
    df[df["Sleep Duration"] < 6]
    df.groupby("BMI Category").agg(
        average_sleep=("Sleep Duration", "mean"),
        average_quality=("Quality of Sleep", "mean"),
        record_count=("Person ID", "count"),
    )


def polars_analysis():
    pl_df.group_by("Stress Level").agg(pl.col("Quality of Sleep").mean())
    pl_df.filter(pl.col("Stress Level") >= 7)
    pl_df.filter(pl.col("Stress Level") <= 4)
    pl_df.filter(pl.col("Sleep Duration") < 6)
    pl_df.group_by("BMI Category").agg(
        pl.col("Sleep Duration").mean(),
        pl.col("Quality of Sleep").mean(),
        pl.len(),
    )


# One timing swings a lot on 374 rows, so average over repeats.
REPEATS = 100

start = time.perf_counter()
for _ in range(REPEATS):
    pandas_analysis()
pandas_time = (time.perf_counter() - start) / REPEATS

start = time.perf_counter()
for _ in range(REPEATS):
    polars_analysis()
polars_time = (time.perf_counter() - start) / REPEATS

print(f"Average Pandas analysis time: {pandas_time * 1000:.3f} ms")
print(f"Average Polars analysis time: {polars_time * 1000:.3f} ms")

if pandas_time < polars_time:
    print("Pandas was faster in this small test.")
else:
    print("Polars was faster in this small test.")

print("The dataset has only 374 rows, so this is practice rather than a real benchmark.")


# --- Machine learning ---

section("8. MACHINE LEARNING")

FEATURES = [ # features, x
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic",
    "Diastolic",
]
TARGET = "Quality of Sleep" # label, y


def run_regression(data, label):
    """Fit a linear regression and print how well it did."""
    X = data[FEATURES]
    y = data[TARGET]

    # random_state keeps the split identical between runs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{label} (n={len(data)})")
    print("Mean absolute error:", round(mean_absolute_error(y_test, predictions), 2))
    print("R-squared:", round(r2_score(y_test, predictions), 2))


# I chose linear regression as a simple first model because my target
# and the features I selected are numeric.
run_regression(df, "All rows")
run_regression(unique_profiles, "Repeated profiles removed")

# FINDING 3: R-squared dropped from about 0.93 to 0.90 after removing
# repeated profiles, so the repeated rows are not the main reason the
# model performs well. Stress Level has a strong negative relationship with Quality of Sleep in this dataset.

print("\nCorrelation between stress and sleep quality:")
print(df[["Stress Level", "Quality of Sleep"]].corr().round(2))


# --- Visualisation ---

section("9. VISUALISATION")

# Main plot. Stress Level has few groups, so a bar chart compares their
# average sleep quality directly, and it matches the question above.
stress_summary.plot(kind="bar", figsize=(8, 5))
plt.title("Average Sleep Quality by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Average Quality of Sleep")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("images/average_sleep_quality_by_stress.png", dpi=150)
plt.close()

# Stacked bars show group size and the disorder split inside each group.
pd.crosstab(df["BMI Category"], df["Sleep Disorder"]).plot(
    kind="bar", stacked=True, figsize=(9, 5)
)
plt.title("Sleep Disorders by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.legend(title="Sleep Disorder")
plt.tight_layout()
plt.savefig("images/bmi_vs_sleep_disorder.png", dpi=150)
plt.close()

# Same relationship as the main plot, but showing spread within each level.
sns.boxplot(data=df, x="Stress Level", y="Quality of Sleep")
plt.title("Sleep Quality by Stress Level")
plt.xlabel("Stress Level (1-10)")
plt.ylabel("Quality of Sleep (1-10)")
plt.tight_layout()
plt.savefig("images/stress_vs_quality.png", dpi=150)
plt.close()

print("\nPlots saved to the images/ folder.")
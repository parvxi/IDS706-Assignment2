import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import EDA_Dataset as eda

CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "Sleep_health_and_lifestyle_dataset.csv"
)


def test_load_data_shape():
    print("\n" + "-" * 50)
    print("📟 📟 📟 Loading the data in progress 📟 📟 📟")
    print("-" * 50)
    df = eda.load_data(CSV)
    assert df.shape == (374, 13)


def test_clean_data_fixes_values():
    print("\n" + "-" * 50)
    print("🧼 🧼 🧼 Cleaning the data in progress 🧼 🧼 🧼")
    print("-" * 50)
    raw = eda.load_data(CSV)
    df = eda.clean_data(raw)

    # 1. Sleep Disorder should have NO missing values left
    assert df["Sleep Disorder"].isnull().sum() == 0
    assert df["Sleep Disorder"].notnull().all()
    assert "Normal Weight" not in df["BMI Category"].values


def test_load_data_missing_file():
    print("\n" + "-" * 50)
    print("🚫 🚫 🚫 Loading a file that does not exist 🚫 🚫 🚫")
    print("\n" + "-" * 50)
    with pytest.raises(FileNotFoundError):
        eda.load_data("data/this_file_does_not_exist.csv")


def test_clean_data_small_example():
    """Checks if each cleaning step do exactly the right thing?"""
    print("\n" + "-" * 50)
    print("📊 📊 📊 Testing with sample data if the cleaning is correct 📊 📊 📊")
    print("-" * 50)
    raw = pd.DataFrame(
        {
            "Sleep Disorder": [None, "Insomnia"],
            "BMI Category": ["Normal Weight", "Obese"],
            "Blood Pressure": ["126/83", "140/95"],
        }
    )
    df = eda.clean_data(raw)

    assert df["Sleep Disorder"].tolist() == ["No Disorder", "Insomnia"]
    assert df["BMI Category"].tolist() == ["Normal", "Obese"]
    assert df["Systolic"].tolist() == [126, 140]
    assert df["Diastolic"].tolist() == [83, 95]
    assert raw["Sleep Disorder"].isnull().sum() == 1


def test_stress_summary():
    """Checks if the analysis give the same numbers I wrote in the README?"""
    print("\n" + "-" * 50)
    print("😰 😰 😰 Checking sleep quality by stress level 😰 😰 😰")
    print("-" * 50)
    df = eda.clean_data(eda.load_data(CSV))
    summary = eda.stress_summary(df)

    assert round(summary[3], 2) == 8.97
    assert round(summary[8], 2) == 5.86
    assert summary[3] > summary[8]


def test_regression_results():
    print("\n" + "-" * 50)
    print("🤖 🤖 🤖 Testing the model results 🤖 🤖 🤖")
    print("-" * 50)
    df = eda.clean_data(eda.load_data(CSV))
    result = eda.run_regression(df, "test")

    assert 0 <= result["r2"] <= 1
    assert result["mae"] >= 0
    assert len(result["predictions"]) == len(result["y_test"])


def test_regression_is_reproducible():
    print("\n" + "-" * 50)
    print("🤖 🤖 🤖 Testing the model reproducibility 🤖 🤖 🤖")
    print("-" * 50)
    df = eda.clean_data(eda.load_data(CSV))
    first = eda.run_regression(df, "first")
    second = eda.run_regression(df, "second")

    assert first["r2"] == second["r2"]
    assert list(first["predictions"]) == list(second["predictions"])


def test_full_pipeline():
    """System test: run the whole analysis from start to finish."""
    print("\n" + "-" * 50)
    print("🚀 🚀 🚀 Testing the whole pipeline 🚀 🚀 🚀")
    print("-" * 50)
    project_folder = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_folder)

    df = eda.main()

    assert df.shape == (374, 15)
    assert df["Sleep Disorder"].isnull().sum() == 0
    for plot in [
        "average_sleep_quality_by_stress.png",
        "bmi_vs_sleep_disorder.png",
        "stress_vs_quality.png",
    ]:
        assert os.path.exists(os.path.join("images", plot))


# to run write "pytest" in the termianl

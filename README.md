# IDS706 — Week 2 Mini-Assignment: Sleep, Stress & Lifestyle Analysis

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?logo=pandas&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-1.44-CD792C?logo=polars&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-evcxr__jupyter-DEA584?logo=rust&logoColor=white)

<img src="images/stress-image-readme.jpg" alt="Illustration of someone lying awake, stressed and unable to sleep" width="600">

> Does stress actually wreck your sleep, or is that just something we all say? Let's find out with data.

---

## Project Goal

This is **Series 1 of a 3-week mini-project** for IDS706. The goal is to take a beginner-friendly dataset through a full first-pass data analysis workflow: load it, inspect it, clean it, filter/group it, run it through a simple machine learning model, and visualize what's going on — all while documenting the reasoning along the way, since I'll be reusing this same dataset next week for testing, CI, and refactoring.

The question I set out to explore:

> **Which lifestyle factors line up with sleep quality, and can a simple regression predict it?**

---

## Dataset

**[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)** (Kaggle)

- **374 rows** × **13 columns**
- Synthetic dataset (created by the dataset author) covering age, gender, occupation, sleep duration/quality, physical activity, stress level, BMI category, blood pressure, heart rate, daily steps, and sleep disorder
- Stored locally at [`data/Sleep_health_and_lifestyle_dataset.csv`](data/Sleep_health_and_lifestyle_dataset.csv)

---

## Setup & How to Run

**1. Clone the repo and move into it**
```powershell
git clone https://github.com/parvxi/IDS706-Assignment2.git
cd IDS706-Assignment2
```

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```powershell
pip install -r requirements.txt
```

**4. Run the analysis**
```powershell
python EDA_Dataset.py
```

The script prints every step to the terminal and (re)generates the plots inside [`images/`](images/).

---

## Analysis Steps

The script ([`EDA_Dataset.py`](EDA_Dataset.py)) is organized into clearly labeled sections, in order:

| # | Section | What happens |
|---|---------|---------------|
| 1 | **Load** | Read the CSV into a Pandas DataFrame |
| 2 | **Inspect** | `.head()`, `.info()`, `.describe()`, `.nunique()`, missing values, duplicate checks |
| 3 | **Clean** | Fix categories and split the `Blood Pressure` text column into numeric `Systolic`/`Diastolic` |
| 4 | **Explore** | Distribution of sleep disorders and occupations |
| 5 | **Pandas analysis** | Filtering + `groupby()` summary statistics |
| 6 | **Polars analysis** | The *same* filtering/grouping logic re-implemented in Polars |
| 7 | **Pandas vs. Polars** | A small timing comparison between the two libraries |
| 8 | **Machine learning** | A linear regression predicting `Quality of Sleep` |
| 9 | **Visualization** | Three saved plots |

### Inspecting the data

`.info()` and `.describe()` showed 13 columns with no missing values — **except** `Sleep Disorder`, which had 219 nulls out of 374 rows. `.duplicated()` found 0 fully identical rows (every `Person ID` is unique), but **242 rows become duplicates once `Person ID` is dropped**, meaning the dataset really only has 132 distinct lifestyle "profiles" repeated across different people.

### Cleaning decisions

- **`Sleep Disorder` NaNs → `"No Disorder"`** — the dataset's original `"None"` string was auto-interpreted by Pandas as a missing value. Since that's actually a valid category (no disorder), I relabeled it instead of dropping 219 of 374 rows.
- **`"Normal Weight"` merged into `"Normal"`** — same BMI category, just an inconsistent label.
- **`Blood Pressure` ("126/83") split into `Systolic` / `Diastolic`** — so it can be used as numeric input later in the regression.

### Filtering & grouping (the interesting part)

| Group | Records | Avg. Sleep Duration | Avg. Sleep Quality |
|---|---|---|---|
| High stress (level ≥ 7) | 120 | 6.22 hrs | 5.92 / 10 |
| Low stress (level ≤ 4) | 141 | 7.63 hrs | 8.33 / 10 |

Grouping average sleep quality by stress level shows a clear downward trend from **8.97 (stress 3) down to 5.86 (stress 8)**. The correlation between `Stress Level` and `Quality of Sleep` came out to **-0.90** a strong negative relationship.

Sleep by BMI category also stood out:

| BMI Category | Avg. Sleep | Avg. Quality | Records |
|---|---|---|---|
| Normal | 7.39 hrs | 7.64 | 216 |
| Overweight | 6.77 hrs | 6.90 | 148 |
| Obese | 6.96 hrs | 6.40 | 10 |

I also ran every grouping on distinct profiles only (with the 242 repeated rows removed), to make sure the repetition in the dataset wasn't skewing the picture the direction of every result held up.

### 🐼 vs 🐻‍❄️ Pandas vs. Polars

I re-ran the same filters and `groupby`/`group_by` operations in both libraries and timed them (averaged over 100 runs):

| Library | Avg. time per run |
|---|---|
| Pandas | 6.223 ms |
| Polars | **2.184 ms** |

Polars came out ~2.8× faster here. That said, with only 374 rows this is really more of a syntax comparison than a real performance benchmark, Polars advantages tend to show up on much larger datasets.

---

## Machine Learning

**Algorithm:** Linear Regression (`scikit-learn`)

**Target:** `Quality of Sleep`
**Features:** `Age`, `Sleep Duration`, `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `Systolic`, `Diastolic`

I picked linear regression as a simple, interpretable baseline since the target and all selected features are numeric. then I split the data 80/20 for train/test.

| Run | n | MAE | R² |
|---|---|---|---|
| All rows | 374 | 0.27 | **0.93** |
| Repeated profiles removed | 132 | 0.29 | **0.90** |

**Takeaway:** R² only dropped from 0.93 to 0.90 after removing the 242 duplicated profiles, so the strong fit isn't just an artifact of repeated rows it's mostly being driven by how tightly `Stress Level` tracks `Quality of Sleep` (r = -0.90). Since this is a **synthetic** dataset, these patterns describe this dataset well but shouldn't be over-generalized to real-world sleep behavior.

---

## Visualization

**Main plot — Average Sleep Quality by Stress Level**

<img src="images/average_sleep_quality_by_stress.png" alt="Bar chart of average sleep quality by stress level" width="600">

A bar chart was the right call here: `Stress Level` only has 6 distinct values, so comparing their average `Quality of Sleep` directly answers the project's main question at a glance quality drops steadily as stress climbs.

**Supporting plot 1 — Sleep Quality distribution by Stress Level (boxplot)**

<img src="images/stress_vs_quality.png" alt="Boxplot of sleep quality distribution across stress levels" width="600">

The boxplot adds what the bar chart can't show: spread. It confirms the downward trend isn't just about averages the whole distribution shifts down and the values get more consistent (less spread) at higher stress levels.

**Supporting plot 2 — Sleep Disorders by BMI Category**

<img src="images/bmi_vs_sleep_disorder.png" alt="Stacked bar chart of sleep disorders by BMI category" width="600">

A stacked bar chart to explore a second angle: among Normal BMI records, roughly 93% report no sleep disorder. Among Overweight records that flips almost completely, only about 13% have no disorder, with the rest split fairly evenly between Insomnia and Sleep Apnea. A relationship worth digging into further next week.

---

## 🦀 🦀 🦀 Question 2 — Rust & Ownership 🦀 🦀 🦀

Worked through [`notebooks/rust_vs_python_intro.ipynb`](notebooks/rust_vs_python_intro.ipynb) using the `evcxr_jupyter` Rust kernel, completing all the *Your turn* exercises and deliberately triggering compiler errors to see the ownership rules enforced:

| Experiment | What the compiler said |
|---|---|
| Reassigning a plain `let` | `cannot assign twice to immutable variable` |
| Adding `mut`, then reassigning | Compiles and runs |
| `let moved = ratings;` then using `ratings` | `borrow of moved value: ratings` |
| Removing `.clone()` myself | Same move error, reproduced on purpose |
| Mutating a `Vec` while looping over it | `borrow of moved value` — Rust refuses the program Python would run |

What stuck with me is the trade-off. Rust made me say in advance which values could change and who owned each list, and refused to compile when I broke those rules including a loop that Python would have run happily while silently giving the wrong answer. Python is more flexible, but that flexibility is exactly what lets those bugs through. It also made .clone() feel concrete like copying a list is a real cost, and Rust makes you ask for it rather than doing it invisibly.

---

## Repo Structure

```
IDS706-Assignment2/
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── images/
│   ├── stress-image-readme.jpg
│   ├── average_sleep_quality_by_stress.png
│   ├── stress_vs_quality.png
│   └── bmi_vs_sleep_disorder.png
├── notebooks/
│   └── rust_vs_python_intro.ipynb   # Question 2 — Rust ownership exercises
├── EDA_Dataset.py
├── requirements.txt
└── README.md
```

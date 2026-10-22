"""
============================================================
  TASK 02 - SkillCraft Technology Internship
  Titanic Dataset: Data Cleaning & Exploratory Data Analysis (EDA)
============================================================

Objectives:
  1. Load Titanic dataset.
  2. Data Cleaning & Preprocessing (handle missing values, duplicates, redundant features).
  3. Feature Engineering (family_size, is_alone, fare_bin, age_group).
  4. Exploratory Data Analysis (EDA):
     - Univariate Analysis
     - Bivariate & Multivariate Analysis
     - Correlation Matrix
     - Visualizations saved to 'charts' directory
  5. Statistical Analysis & Insights.
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Aesthetic Plot Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {0: "#E63946", 1: "#2A9D8F", "0": "#E63946", "1": "#2A9D8F", "Did Not Survive": "#E63946", "Survived": "#2A9D8F"}

def save_plot(filename):
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {filename}")

def main():
    print("=" * 60)
    print(" 1. LOADING DATASET")
    print("=" * 60)
    
    # Load dataset directly via seaborn
    df = sns.load_dataset("titanic")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nInitial Data Preview:")
    print(df.head())

    print("\n" + "=" * 60)
    print(" 2. DATA INSPECTION & CLEANING")
    print("=" * 60)
    
    print("\nMissing Values Count:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # Drop redundant / leaky columns
    cols_to_drop = ["deck", "alive", "embark_town", "class", "who", "adult_male"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    print(f"\nDropped redundant columns: {cols_to_drop}")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates} duplicate rows. Remaining: {df.shape[0]} rows")

    # Impute missing values
    # Age: Median by Pclass & Sex
    df["age"] = df.groupby(["pclass", "sex"], observed=False)["age"].transform(lambda x: x.fillna(x.median()))
    df["age"] = df["age"].fillna(df["age"].median())

    # Embarked: Mode
    mode_embarked = df["embarked"].mode()[0]
    df["embarked"] = df["embarked"].fillna(mode_embarked)

    # Fare: Median
    if df["fare"].isnull().any():
        df["fare"] = df["fare"].fillna(df["fare"].median())

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print(" 3. FEATURE ENGINEERING")
    print("=" * 60)

    # Family Size & Solo Status
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Categorical Age Groups
    bins = [0, 12, 19, 35, 60, 100]
    labels = ["Child", "Teen", "Adult", "Middle-Aged", "Senior"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # Fare Quartiles
    df["fare_bin"] = pd.qcut(df["fare"], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"])

    # Friendly string labels for charting
    df["survival_label"] = df["survived"].map({0: "Did Not Survive", 1: "Survived"})

    print("Created new features: 'family_size', 'is_alone', 'age_group', 'fare_bin', 'survival_label'")

    print("\n" + "=" * 60)
    print(" 4. GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 60)

    # 1. Overall Survival Rate
    plt.figure(figsize=(7, 5))
    counts = df["survival_label"].value_counts()
    colors = ["#E63946", "#2A9D8F"]
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.05, 0))
    plt.title("Titanic Overall Survival Proportion", fontsize=14, fontweight="bold")
    save_plot("01_overall_survival.png")

    # 2. Survival by Gender
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sex", hue="survival_label", palette={"Did Not Survive": "#E63946", "Survived": "#2A9D8F"})
    plt.title("Survival Count by Gender", fontsize=14, fontweight="bold")
    plt.xlabel("Sex", fontsize=12)
    plt.ylabel("Passenger Count", fontsize=12)
    plt.legend(title="Status")
    save_plot("02_survival_by_gender.png")

    # 3. Survival by Passenger Class
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", palette="crest", errorbar=None)
    plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
    plt.xlabel("Pclass (1st, 2nd, 3rd)", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    save_plot("03_survival_rate_by_class.png")

    # 4. Age Distribution by Survival
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df[df["survived"] == 1]["age"], label="Survived", color="#2A9D8F", fill=True)
    sns.kdeplot(data=df[df["survived"] == 0]["age"], label="Did Not Survive", color="#E63946", fill=True)
    plt.title("Age Distribution by Survival Status", fontsize=14, fontweight="bold")
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    save_plot("04_age_distribution_survival.png")

    # 5. Fare Distribution (Boxplot with log scale)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="survival_label", y="fare", palette={"Did Not Survive": "#E63946", "Survived": "#2A9D8F"})
    plt.yscale("log")
    plt.title("Fare Distribution by Survival (Log Scale)", fontsize=14, fontweight="bold")
    plt.xlabel("Survival Status", fontsize=12)
    plt.ylabel("Fare (Log Scale)", fontsize=12)
    save_plot("05_fare_distribution_survival.png")

    # 6. Family Size vs Survival Rate
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="family_size", y="survived", palette="viridis", errorbar=None)
    plt.title("Survival Rate by Family Size", fontsize=14, fontweight="bold")
    plt.xlabel("Family Size", fontsize=12)
    plt.ylabel("Survival Rate", fontsize=12)
    save_plot("06_survival_by_family_size.png")

    # 7. Survival by Embarkation Port
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="embarked", hue="survival_label", palette={"Did Not Survive": "#E63946", "Survived": "#2A9D8F"})
    plt.title("Survival Count by Port of Embarkation", fontsize=14, fontweight="bold")
    plt.xlabel("Port (C = Cherbourg, Q = Queenstown, S = Southampton)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Status")
    save_plot("07_survival_by_embarkation.png")

    # 8. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
    save_plot("08_correlation_heatmap.png")

    # 9. Comprehensive Multi-Feature Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Class & Sex Interaction
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", palette={"male": "#457B9D", "female": "#E76F51"}, errorbar=None, ax=axes[0, 0])
    axes[0, 0].set_title("Survival Rate by Class & Sex", fontsize=13, fontweight="bold")

    # Subplot 2: Age Group Survival
    sns.barplot(data=df, x="age_group", y="survived", palette="mako", errorbar=None, ax=axes[0, 1])
    axes[0, 1].set_title("Survival Rate by Age Group", fontsize=13, fontweight="bold")

    # Subplot 3: Fare Bin Survival
    sns.barplot(data=df, x="fare_bin", y="survived", palette="rocket", errorbar=None, ax=axes[1, 0])
    axes[1, 0].set_title("Survival Rate by Fare Bin", fontsize=13, fontweight="bold")

    # Subplot 4: Solo Travelers vs With Family
    sns.barplot(data=df, x="is_alone", y="survived", palette=["#2A9D8F", "#E63946"], errorbar=None, ax=axes[1, 1])
    axes[1, 1].set_title("Survival Rate: Traveling Alone vs With Family", fontsize=13, fontweight="bold")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["With Family", "Alone"])

    save_plot("09_eda_comprehensive_dashboard.png")

    print("\n" + "=" * 60)
    print(" 5. STATISTICAL TESTS & KEY INSIGHTS")
    print("=" * 60)

    # Chi-Square Test: Sex vs Survived
    contingency_sex = pd.crosstab(df["sex"], df["survived"])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency_sex)
    print(f"Chi-Square Test (Sex vs Survived): Chi2 = {chi2:.2f}, p-value = {p_val:.4e}")

    # Chi-Square Test: Pclass vs Survived
    contingency_pclass = pd.crosstab(df["pclass"], df["survived"])
    chi2_p, p_val_p, _, _ = stats.chi2_contingency(contingency_pclass)
    print(f"Chi-Square Test (Pclass vs Survived): Chi2 = {chi2_p:.2f}, p-value = {p_val_p:.4e}")

    print("\nSUMMARY OF KEY FINDINGS:")
    print("-" * 50)
    print("1. Gender: Females had a significantly higher survival rate (~74%) than males (~19%).")
    print("2. Socio-economic Class: 1st Class passengers had highest survival (~63%), compared to 3rd Class (~24%).")
    print("3. Age: Children had elevated survival odds compared to working-age adults.")
    print("4. Family Size: Passengers travelling with small families (2-4 members) had better survival rates than solo passengers or large families.")
    print("5. Fare: Higher fare strongly positively correlates with survival.")
    print("=" * 60)
    print(f"ALL TASKS COMPLETED! All plots saved to: {CHARTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

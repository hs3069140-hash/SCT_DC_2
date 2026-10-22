# Task 02: Titanic Exploratory Data Analysis (EDA) & Data Cleaning

Internship Task completed for **SkillCraft Technology**.

This repository contains an end-to-end Python pipeline that performs comprehensive data cleaning, feature engineering, exploratory data analysis (univariate, bivariate, and multivariate), and hypothesis testing on the Titanic dataset.

---

## 📁 Project Structure

```
titanic_eda/
│
├── titanic.csv                 # Raw Titanic dataset (standalone copy)
├── titanic_eda.py              # Main Python script (cleaning, EDA, visualizations)
├── requirements.txt            # Python library dependencies
├── run.bat                     # 1-Click launcher for Windows
├── run.sh                      # 1-Click launcher for Linux / macOS
├── README.md                   # Project documentation
│
└── charts/                     # Generated high-resolution plots
    ├── 01_overall_survival.png
    ├── 02_survival_by_gender.png
    ├── 03_survival_rate_by_class.png
    ├── 04_age_distribution_survival.png
    ├── 05_fare_distribution_survival.png
    ├── 06_survival_by_family_size.png
    ├── 07_survival_by_embarkation.png
    ├── 08_correlation_heatmap.png
    └── 09_eda_comprehensive_dashboard.png
```

---

## 🚀 How to Run Anywhere

You can move or copy this entire folder anywhere (USB drive, another computer, GitHub, etc.) and run it seamlessly.

### Option 1: One-Click Execution (Windows)
Double-click [`run.bat`](file:///C:/Users/hs306/.gemini/antigravity-ide/scratch/titanic_eda/run.bat). It will automatically install packages, run the analysis, and open the charts directory.

### Option 2: Linux / macOS
Run the shell script in terminal:
```bash
chmod +x run.sh
./run.sh
```

### Option 3: Manual Command Line
1. Open terminal inside this folder:
   ```bash
   pip install -r requirements.txt
   python titanic_eda.py
   ```

---

## 📊 Summary of Insights

1. **Gender:** Female survival rate was ~74% compared to ~19% for males ($\chi^2 = 203.61, p < 10^{-45}$).
2. **Socio-Economic Class:** 1st Class passengers had ~63% survival rate, 2nd Class ~47%, and 3rd Class only ~24% ($\chi^2 = 90.00, p < 10^{-19}$).
3. **Age Groups:** Children ($\le 12$ years) had higher survival rates than adults.
4. **Family Size:** Small families (2 to 4 members) had higher survival rates than solo travelers or large families ($>5$).
5. **Fare:** Ticket price positively correlated with survival.

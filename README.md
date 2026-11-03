# Analysis of Factors Associated with Diabetes Status
Capstone Mini Project · Python for Data Science · S.Y. B.Tech EXCP · Healthcare domain
Dataset: CDC BRFSS 2015 – *Diabetes Health Indicators* (Kaggle, Alex Teboul) – our file `diabetesData.csv` = the **50/50 balanced** version: 70,692 rows × 22 columns (35,346 with and 35,346 without diabetes), no missing values, 1,635 duplicate rows. After cleaning: 69,057 rows × 35 columns.

## 1. How to run
```
pip install pandas numpy matplotlib seaborn scipy statsmodels
```
`statsmodels` is **required**: `statistical_analysis.py` stops with a clear message if it is missing.
1. Put `diabetesData.csv` in the `data/` folder (already included in the zip).
2. Run `python main.py` and use the menu (option **7** runs everything), or `python main.py --auto`.
3. Everything is created in `outputs/`:

| File | Content |
|---|---|
| `cleaned_diabetes_data.csv` | the cleaned dataset (this is the "cleaned CSV" deliverable) |
| `cleaning_log.txt` | every cleaning step with counts |
| `figures/01…18_*.png` | 18 charts |
| `statistical_results.csv`, `chi_square_results.csv`, `odds_ratios.csv`, `logistic_regression.csv`, `ols_regression.csv` | test results (summary and chi-square tables include Holm / FDR adjusted p-values) |
| `analysis_report.txt` | auto-generated summary of findings |

## 2. Project structure
| File | Role | Syllabus |
|---|---|---|
| `main.py` | menu-driven app, report builder | Unit 1.1–1.4, 2.1 (`os`, `time`, `datetime`, `sys`) |
| `database_manipulation.py` | load, validate, impute, de-duplicate, outliers, feature engineering, group-by tables | Unit 1.2, 2.1 (`re`, `os`), 2.2, 3.1, 3.2 |
| `visualization.py` | 18 charts | Unit 4.1, 4.2 |
| `statistical_analysis.py` | SciPy tests; Statsmodels Tukey HSD, OLS + `anova_lm`, Holm/FDR correction, logistic regression | Module 5 (SciPy / Statsmodels) |

`main.py` imports the other three files – this *is* the **user-defined module** topic (Unit 2.2).

## 3. Rubric → where it is satisfied
| Criterion (weight) | Evidence |
|---|---|
| Problem definition & dataset (10%) | Healthcare theme; clear question; 253k rows × 22 columns; CSV + source link |
| Code structure, modularity (15%) | 4 modules, 40+ small functions, docstrings, try/except, menu with `while` loop |
| Cleaning & manipulation – NumPy/Pandas (20%) | range validation, imputation, `drop_duplicates`, `astype`, IQR outliers (`np.percentile`, `np.clip`), `map`, `pd.cut`, `np.where`, `sum(axis=1)`, `groupby/agg`, `loc/iloc/query`, `sort_values` |
| Visualisation – Matplotlib/Seaborn (20%) | pie, bar, histogram, line+area, subplots; box, violin, count, bar, point, line, catplot, FacetGrid, regplot, joint hexbin, pairplot, heatmap |
| Statistics – SciPy/Statsmodels (20%) | normality, Levene/Bartlett, Pearson/Spearman/Kendall, chi-square (+Cramér's V), 1-sample/independent/paired t-test, ANOVA (SciPy) + Tukey HSD (statsmodels), one-/two-way ANOVA tables and OLS regression (statsmodels), Holm/FDR multiple-testing correction, Mann-Whitney, Wilcoxon, Kruskal-Wallis, KS, odds ratios, logistic regression |
| Report & interpretation (10%) | `analysis_report.txt` + effect sizes + limitations |
| Viva (5%) | see section 5 |

## 4. Project design decisions (be ready to justify)
* **Target**: `Diabetes_binary` (1 = prediabetes or diabetes, 0 = none) → cleaned into `Diabetes_Binary` plus the label `Diabetes_Status` (No Diabetes / Diabetes). The raw file has only these two classes, so ANOVA/Kruskal-Wallis run with two groups.
* **Balanced sample**: the file is 50/50 by design, so the percentages in charts are the *share with diabetes inside this sample*, NOT real population prevalence. Differences between groups, odds ratios and tests are still valid.
* **Duplicates removed**: identical rows add no information in a survey with only categorical answers.
* **BMI outliers flagged, not deleted**: values are medically possible; `BMI_Capped` (IQR fences) is used for parametric tests, raw BMI for non-parametric ones.
* **Normality fails → both** parametric and non-parametric tests are run and agree.
* **Why effect sizes?** With 250k rows nearly every p-value is < 0.05, so Cramér's V, Cohen's d and odds ratios show *importance*.
* **Why Tukey HSD?** After a significant ANOVA with 3+ groups (PhysHlth across the 4 BMI categories) it compares every pair while keeping the overall error rate at 5%; statsmodels' `pairwise_tukeyhsd` is used.
* **Why correct for multiple testing?** Running ~60 tests makes some false positives likely. Holm (controls family-wise error) and Benjamini-Hochberg (controls false discovery rate) from `statsmodels.stats.multitest.multipletests` adjust the p-values.
* **OLS / ANOVA tables**: `smf.ols` + `anova_lm` give one-way and two-way ANOVA (Diabetes status x Sex) and a multiple linear regression for BMI.
* **Forecasting** (rubric: "Statistical analysis / Forecasting") does not apply: the data is a cross-sectional survey with no time axis, so statistical analysis is used.
* **Association ≠ causation**: stated in limitations.

## 5. Likely viva questions
1. *Why this dataset / what is the problem?* – diabetes is a major public-health issue; the survey links lifestyle, health and socio-economic factors to diabetes status.
2. *Which functions did you use to clean?* – `isna().sum()`, `fillna()`, `drop_duplicates()`, `astype()`, `np.percentile()`, `np.clip()`, `map()`, `pd.cut()`.
3. *Why median for missing numbers and mode for binary columns?* – median is robust to outliers; mode keeps 0/1 values valid.
4. *Difference between `loc` and `iloc`?* – label-based vs position-based (used in `high_risk_group` and `print_manipulation_demo`).
5. *Why violin plot as well as box plot?* – violin also shows the density shape.
6. *What does the chi-square test check?* – H0: two categorical variables are independent; p < 0.05 → association.
7. *Why Welch t-test?* – group variances/sizes are unequal (Levene), Welch does not assume equal variance.
8. *Why Mann-Whitney U?* – data not normal; compares distributions via ranks.
9. *What is an odds ratio?* – odds of diabetes with the factor ÷ odds without it; OR > 1 raises risk.
10. *Why AUC, not accuracy?* – accuracy depends on the cut-off and the class mix; AUC (0.82 in our model) measures how well the model ranks diabetic above non-diabetic people.
13. *Why does heavy alcohol look protective and Cholesterol Check look like a risk?* – reverse causation / screening bias: diagnosed people are told to avoid alcohol and are tested more often. Association ≠ causation.
14. *Why is the data 50/50?* – it is a balanced subsample made for modelling; so we compare groups but do not quote prevalence.
11. *What is a user-defined module?* – our own `.py` file imported with `import database_manipulation as dm`.
12. *Limitations?* – self-reported, cross-sectional, balanced (not population) sample, no causation.

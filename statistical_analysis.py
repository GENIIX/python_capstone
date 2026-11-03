"""
statistical_analysis.py
=======================
Statistical analysis of the diabetes data (USER-DEFINED MODULE) - Module 5 (SciPy / Statsmodels).
SciPy runs the classical hypothesis tests; Statsmodels runs the post-hoc, model-based and multiple-testing parts.

Tests used (all taught in Module 5):
  * Descriptive statistics     : mean, median, mode, variance, std, skewness, kurtosis, percentiles
  * Normality                  : Shapiro-Wilk, D'Agostino-Pearson K2, Anderson-Darling, Kolmogorov-Smirnov
  * Homogeneity of variance    : Levene, Bartlett
  * Correlation                : Pearson, Spearman, Kendall
  * Categorical association    : Chi-square test of independence (+ Cramer's V)
  * Parametric tests           : one-sample t, independent t (Welch), paired t, one-way ANOVA (SciPy)
  * Statsmodels                : Tukey HSD post-hoc, OLS regression, ANOVA tables (anova_lm, one- and two-way),
                                 multiple-testing correction (Holm, Benjamini-Hochberg FDR), logistic regression
  * Non-parametric tests       : Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, two-sample KS
Extension : Odds ratios (2x2 tables) and multivariable logistic regression (statsmodels).
Requires : pip install scipy statsmodels

Significance level used everywhere: alpha = 0.05
"""

import math
import warnings

import sys

import numpy as np
import pandas as pd
from scipy import stats

try:                                                                 # statsmodels is REQUIRED (core of this module)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
except ImportError:
    sys.exit("[ERROR] statsmodels is required. Install it with:  pip install statsmodels")

from database_manipulation import BINARY_COLUMNS, RISK_FACTOR_LABELS

ALPHA = 0.05
SEED = 42


# ----------------------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------------------
def decision(p_value, alpha=ALPHA):
    """Hypothesis-test decision text (if-else)."""
    if p_value < alpha:
        return "Reject H0 (significant)"
    return "Fail to reject H0 (not significant)"


def format_p(p_value):
    """Print very small p-values as '<0.001'."""
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def effect_label(value, kind="r"):
    """Describe the size of an effect using an if-elif-else ladder (Unit 1.3)."""
    value = abs(value)
    limits = {"d": (0.2, 0.5, 0.8), "r": (0.1, 0.3, 0.5), "v": (0.1, 0.3, 0.5), "eta": (0.01, 0.06, 0.14)}
    small, medium, large = limits[kind]
    if value < small:
        return "negligible"
    elif value < medium:
        return "small"
    elif value < large:
        return "medium"
    return "large"


def _row(test, variables, statistic, p_value, effect="", note="", decision_text=None):
    """One result row of the master summary table."""
    return {
        "Test": test,
        "Variables": variables,
        "Statistic": round(float(statistic), 4),
        "P_Value": p_value,
        "Decision": decision_text if decision_text else decision(p_value),
        "Effect_Size": effect,
        "Interpretation": note,
    }


def _print_rows(rows):
    for row in rows:
        print(f"  {row['Test']:<34} {row['Variables']:<34} stat={row['Statistic']:>12,.4f}  "
              f"p={format_p(row['P_Value']):<8} {row['Decision']}")
        if row["Effect_Size"] or row["Interpretation"]:
            print(f"      -> {row['Effect_Size']} {row['Interpretation']}".rstrip())


def _split_groups(df, value_col, group_col="Diabetes_Binary"):
    """Return {group_value: NumPy array of value_col} (dictionary + loop)."""
    return {g: df.loc[df[group_col] == g, value_col].to_numpy(dtype=float) for g in sorted(df[group_col].unique())}


# ----------------------------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# ----------------------------------------------------------------------------------
def descriptive_statistics(df, columns=("BMI", "MentHlth", "PhysHlth", "Age", "Income", "Risk_Factor_Count")):
    """Central tendency, spread and shape for each column (scipy.stats + NumPy)."""
    rows = []
    for col in columns:
        x = df[col].to_numpy(dtype=float)
        rows.append({
            "Variable": col,
            "Mean": np.mean(x),
            "Median": np.median(x),
            "Mode": df[col].mode()[0],
            "Std": np.std(x, ddof=1),
            "Variance": np.var(x, ddof=1),
            "Min": x.min(),
            "Max": x.max(),
            "Q1": np.percentile(x, 25),
            "Q3": np.percentile(x, 75),
            "Skewness": stats.skew(x),
            "Kurtosis": stats.kurtosis(x),
        })
    return pd.DataFrame(rows).round(3)


# ----------------------------------------------------------------------------------
# 2. NORMALITY & HOMOGENEITY OF VARIANCE
# ----------------------------------------------------------------------------------
def normality_tests(df, col="BMI"):
    """H0: the variable is normally distributed."""
    x = df[col].dropna()
    sample = x.sample(n=min(5000, len(x)), random_state=SEED)       # Shapiro-Wilk is valid up to ~5000 values
    rows = []

    stat, p = stats.shapiro(sample)
    rows.append(_row("Shapiro-Wilk (n=5000 sample)", col, stat, p))

    stat, p = stats.normaltest(x)
    rows.append(_row("D'Agostino-Pearson K2", col, stat, p))

    with warnings.catch_warnings():                                  # hide version warnings of SciPy
        warnings.simplefilter("ignore")
        ad = stats.anderson(sample, dist="norm")
    index_5pct = list(ad.significance_level).index(5.0)
    critical = ad.critical_values[index_5pct]
    text = "Reject H0 (stat > critical value)" if ad.statistic > critical else "Fail to reject H0"
    rows.append(_row("Anderson-Darling (5% critical)", col, ad.statistic, float("nan"),
                     note=f"critical value = {critical:.3f}", decision_text=text))

    standardised = (sample - sample.mean()) / sample.std(ddof=1)
    stat, p = stats.kstest(standardised, "norm")
    rows.append(_row("Kolmogorov-Smirnov vs Normal", col, stat, p))
    return rows


def homogeneity_tests(df, col="BMI"):
    """H0: both diabetes groups have equal variance of col."""
    groups = list(_split_groups(df, col).values())
    rows = []
    stat, p = stats.levene(*groups, center="median")
    rows.append(_row("Levene (median-centred)", f"{col} by Diabetes_Binary", stat, p))
    levene_p = p
    stat, p = stats.bartlett(*groups)
    rows.append(_row("Bartlett", f"{col} by Diabetes_Binary", stat, p))
    return rows, levene_p


# ----------------------------------------------------------------------------------
# 3. CORRELATION
# ----------------------------------------------------------------------------------
def correlation_tests(df):
    """Pearson, Spearman and Kendall for selected pairs of continuous / ordinal variables."""
    pairs = [("BMI", "PhysHlth"), ("BMI", "MentHlth"), ("PhysHlth", "MentHlth"),
             ("Age", "Income"), ("GenHlth", "PhysHlth"), ("Income", "Education")]
    sample = df.sample(n=min(3000, len(df)), random_state=SEED)      # Kendall is slow on big data
    rows = []
    for a, b in pairs:
        r, p = stats.pearsonr(df[a], df[b])
        rows.append(_row("Pearson r", f"{a} vs {b}", r, p, effect_label(r, "r"),
                         "linear relation, " + ("positive" if r > 0 else "negative")))
        rho, p = stats.spearmanr(df[a], df[b])
        rows.append(_row("Spearman rho", f"{a} vs {b}", rho, p, effect_label(rho, "r"),
                         "monotonic relation, " + ("positive" if rho > 0 else "negative")))
        tau, p = stats.kendalltau(sample[a], sample[b])
        rows.append(_row("Kendall tau (n=3000)", f"{a} vs {b}", tau, p, effect_label(tau, "r"), ""))
    return rows


def target_correlations(df):
    """Spearman correlation of every predictor with Diabetes_Binary, ranked by strength."""
    predictors = (["BMI", "Age", "Income", "Education", "GenHlth", "PhysHlth", "MentHlth", "Risk_Factor_Count"]
                  + BINARY_COLUMNS)
    rows = []
    for col in predictors:
        rho, p = stats.spearmanr(df[col], df["Diabetes_Binary"])
        rows.append({"Variable": col, "Spearman_rho": round(rho, 4), "P_Value": p, "Strength": effect_label(rho, "r")})
    table = pd.DataFrame(rows)
    table["Abs"] = table["Spearman_rho"].abs()
    return table.sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)


# ----------------------------------------------------------------------------------
# 4. CHI-SQUARE TEST OF INDEPENDENCE
# ----------------------------------------------------------------------------------
def chi_square_tests(df):
    """H0: the factor and diabetes status are independent. Cramer's V gives the strength."""
    factors = BINARY_COLUMNS + ["Age", "Education", "Income", "GenHlth", "BMI_Category"]
    n = len(df)
    rows = []
    for factor in factors:
        table = pd.crosstab(df[factor], df["Diabetes_Binary"])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        k = min(table.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * k))
        rows.append({"Factor": factor, "Chi2": round(chi2, 2), "DoF": dof, "P_Value": p,
                     "Cramers_V": round(cramers_v, 4), "Strength": effect_label(cramers_v, "v"),
                     "Min_Expected": round(expected.min(), 1), "Decision": decision(p)})
    result = pd.DataFrame(rows).sort_values("Cramers_V", ascending=False).reset_index(drop=True)
    return result


# ----------------------------------------------------------------------------------
# 5. PARAMETRIC TESTS
# ----------------------------------------------------------------------------------
def cohens_d(a, b):
    """Standardised mean difference using the pooled standard deviation."""
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def one_sample_ttest(df, col="BMI_Capped", popmean=25.0):
    """H0: the mean BMI of people WITH diabetes equals 25 (upper limit of normal BMI)."""
    sample = df.loc[df["Diabetes_Binary"] == 1, col].to_numpy(dtype=float)
    t, p = stats.ttest_1samp(sample, popmean)
    note = f"mean = {sample.mean():.2f} vs {popmean}"
    return [_row("One-sample t-test", f"{col} (diabetes group) vs {popmean}", t, p, note=note)]


def independent_ttest(df, col="BMI_Capped", equal_var=False):
    """H0: mean BMI is the same for people with and without diabetes (Welch if variances differ)."""
    groups = _split_groups(df, col)
    no, yes = groups[0], groups[1]
    t, p = stats.ttest_ind(yes, no, equal_var=equal_var)
    d = cohens_d(yes, no)
    name = "Independent t-test (Student)" if equal_var else "Independent t-test (Welch)"
    note = f"mean diff = {yes.mean() - no.mean():.2f} BMI points ({yes.mean():.2f} vs {no.mean():.2f})"
    return [_row(name, f"{col} by Diabetes_Binary", t, p, f"Cohen's d = {d:.2f} ({effect_label(d, 'd')})", note)]


def paired_ttest(df, a="PhysHlth", b="MentHlth"):
    """H0: the same person reports equal average days of poor physical and poor mental health."""
    t, p = stats.ttest_rel(df[a], df[b])
    diffs = df[a].to_numpy(dtype=float) - df[b].to_numpy(dtype=float)
    d = diffs.mean() / diffs.std(ddof=1)
    note = f"mean {a} = {df[a].mean():.2f} days, mean {b} = {df[b].mean():.2f} days"
    return [_row("Paired t-test", f"{a} vs {b}", t, p, f"d_z = {d:.2f} ({effect_label(d, 'd')})", note)]


def one_way_anova(df, col="BMI_Capped", group_col="Diabetes_Status"):
    """H0: mean of col is equal in all groups. Effect size = eta squared (NumPy)."""
    names = sorted(df[group_col].unique())
    arrays = [df.loc[df[group_col] == name, col].to_numpy(dtype=float) for name in names]
    f_value, p = stats.f_oneway(*arrays)

    all_values = np.concatenate(arrays)
    grand_mean = all_values.mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total
    means = ", ".join(f"{n}: {a.mean():.2f}" for n, a in zip(names, arrays))
    rows = [_row("One-way ANOVA", f"{col} by {group_col}", f_value, p,
                 f"eta^2 = {eta2:.3f} ({effect_label(eta2, 'eta')})", means)]

    # Tukey HSD post-hoc from statsmodels (works for any number of groups, gives adjusted p-values)
    data = pd.DataFrame({"value": all_values, "group": np.concatenate([[n] * len(a) for n, a in zip(names, arrays)])})
    tukey = pairwise_tukeyhsd(endog=data["value"], groups=data["group"], alpha=ALPHA)
    table = tukey.summary().data                                     # header row + one row per pair of groups
    for g1, g2, diff, p_adj, low, high, reject in table[1:]:
        rows.append(_row("Tukey HSD post-hoc (statsmodels)", f"{g1} vs {g2}", float(diff), float(p_adj),
                         note=f"95% CI [{float(low):.2f}, {float(high):.2f}]",
                         decision_text="Reject H0 (significant)" if reject else "Fail to reject H0 (not significant)"))
    return rows


# ----------------------------------------------------------------------------------
# 6. NON-PARAMETRIC TESTS
# ----------------------------------------------------------------------------------
def mann_whitney_tests(df, columns=("BMI", "PhysHlth", "MentHlth", "Income")):
    """H0: both diabetes groups have the same distribution. Effect = probability of superiority."""
    rows = []
    for col in columns:
        groups = _split_groups(df, col)
        no, yes = groups[0], groups[1]
        u, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
        prob_superiority = u / (len(yes) * len(no))                  # P(random diabetic > random non-diabetic)
        r = 2 * prob_superiority - 1                                 # rank-biserial correlation
        note = (f"P(diabetes > no diabetes) = {prob_superiority:.3f}; medians {np.median(yes):.1f} vs {np.median(no):.1f}")
        rows.append(_row("Mann-Whitney U", f"{col} by Diabetes_Binary", u, p, f"rank-biserial r = {r:.2f} ({effect_label(r, 'r')})", note))
    return rows


def wilcoxon_test(df, a="PhysHlth", b="MentHlth"):
    """H0: median difference between paired measurements is zero (random sample of 5000)."""
    sample = df.sample(n=min(5000, len(df)), random_state=SEED)
    res = stats.wilcoxon(sample[a], sample[b])
    return [_row("Wilcoxon signed-rank (n=5000)", f"{a} vs {b}", res.statistic, res.pvalue)]


def kruskal_wallis_test(df, col="PhysHlth", group_col="Diabetes_Status"):
    """H0: all groups come from the same distribution."""
    names = sorted(df[group_col].unique())
    arrays = [df.loc[df[group_col] == n, col].to_numpy(dtype=float) for n in names]
    h, p = stats.kruskal(*arrays)
    medians = ", ".join(f"{n}: {np.median(a):.1f}" for n, a in zip(names, arrays))
    return [_row("Kruskal-Wallis H", f"{col} by {group_col}", h, p, note=f"medians -> {medians}")]


def ks_two_sample(df, col="BMI"):
    """H0: BMI of the two diabetes groups follows the same distribution."""
    groups = _split_groups(df, col)
    res = stats.ks_2samp(groups[1], groups[0])
    return [_row("Kolmogorov-Smirnov 2-sample", f"{col} by Diabetes_Binary", res.statistic, res.pvalue)]


# ----------------------------------------------------------------------------------
# 6b. STATSMODELS: ANOVA TABLES, OLS REGRESSION, MULTIPLE-TESTING CORRECTION
# ----------------------------------------------------------------------------------
def multi_group_anova(df, col="PhysHlth", group_col="BMI_Category"):
    """One-way ANOVA with a Tukey HSD post-hoc on a variable with MORE than two groups (BMI categories)."""
    return one_way_anova(df, col, group_col)


def anova_tables(df):
    """
    Statsmodels ANOVA tables (type II sums of squares) from OLS models:
      * one-way : BMI_Capped ~ Diabetes_Status
      * two-way : BMI_Capped ~ Diabetes_Status * Sex_Label  (main effects + interaction)
    Returns (rows for the summary table, dictionary of ANOVA DataFrames).
    """
    tables, rows = {}, []
    models = {"One-way ANOVA (OLS)": "BMI_Capped ~ C(Diabetes_Status)",
              "Two-way ANOVA (OLS)": "BMI_Capped ~ C(Diabetes_Status) * C(Sex_Label)"}
    for name, formula in models.items():
        model = smf.ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)
        tables[name] = table
        total_ss = table["sum_sq"].sum()
        for term, r in table.iterrows():
            if term == "Residual":
                continue
            eta2 = r["sum_sq"] / total_ss                            # share of variance explained by the term
            rows.append(_row(name, term.replace("C(", "").replace(")", ""), r["F"], float(r["PR(>F)"]),
                             f"eta^2 = {eta2:.3f} ({effect_label(eta2, 'eta')})"))
    return rows, tables


def ols_regression(df):
    """
    Multiple linear regression (statsmodels OLS): which factors explain BMI?
    Returns (coefficient table, model info dictionary).
    """
    formula = "BMI_Capped ~ Age + Income + Education + PhysHlth + MentHlth + C(Diabetes_Status) + C(Sex_Label)"
    model = smf.ols(formula, data=df).fit()
    conf = model.conf_int()
    table = pd.DataFrame({
        "Predictor": model.params.index,
        "Coefficient": model.params.values.round(4),
        "CI_Lower": conf[0].values.round(4),
        "CI_Upper": conf[1].values.round(4),
        "P_Value": model.pvalues.values,
    })
    table = table[table["Predictor"] != "Intercept"].reset_index(drop=True)
    info = {"n": int(model.nobs), "r2": model.rsquared, "adj_r2": model.rsquared_adj,
            "f_p": model.f_pvalue, "formula": formula}
    return table, info


def correct_multiple_tests(p_values, alpha=ALPHA):
    """Holm (family-wise error) and Benjamini-Hochberg (false discovery rate) adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    holm = multipletests(p, alpha=alpha, method="holm")
    fdr = multipletests(p, alpha=alpha, method="fdr_bh")
    return holm[1], holm[0], fdr[1], fdr[0]


def add_multiple_testing_columns(table, p_col="P_Value"):
    """Add Holm / FDR adjusted p-values to any table that has a P_Value column (NaN p-values are skipped)."""
    table = table.copy()
    valid = table[p_col].notna()
    holm_p, holm_rej, fdr_p, fdr_rej = correct_multiple_tests(table.loc[valid, p_col])
    table["P_Holm"], table["P_FDR"] = np.nan, np.nan
    table["Sig_After_Holm"] = ""
    table.loc[valid, "P_Holm"] = holm_p
    table.loc[valid, "P_FDR"] = fdr_p
    table.loc[valid, "Sig_After_Holm"] = np.where(holm_rej, "yes", "no")
    return table


# ----------------------------------------------------------------------------------
# 7. ODDS RATIOS & LOGISTIC REGRESSION (extension)
# ----------------------------------------------------------------------------------
def odds_ratios(df):
    """Odds ratio (+95% CI) of diabetes for every yes/no factor from its 2x2 table."""
    rows = []
    for column, label in RISK_FACTOR_LABELS.items():
        yes = df[df[column] == 1]["Diabetes_Binary"]
        no = df[df[column] == 0]["Diabetes_Binary"]
        a, b = int(yes.sum()), int((yes == 0).sum())                 # exposed: diabetes / no diabetes
        c, d = int(no.sum()), int((no == 0).sum())                   # not exposed
        if min(a, b, c, d) == 0:                                     # Haldane correction avoids dividing by zero
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        odds_ratio = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        low = math.exp(math.log(odds_ratio) - 1.96 * se)
        high = math.exp(math.log(odds_ratio) + 1.96 * se)
        relative_risk = (a / (a + b)) / (c / (c + d))
        rows.append({"Factor": label, "Odds_Ratio": round(odds_ratio, 2), "CI_Lower": round(low, 2),
                     "CI_Upper": round(high, 2), "Relative_Risk": round(relative_risk, 2),
                     "Meaning": "raises risk" if low > 1 else ("lowers risk" if high < 1 else "no clear effect")})
    return pd.DataFrame(rows).sort_values("Odds_Ratio", ascending=False).reset_index(drop=True)


def logistic_regression(df):
    """
    Multivariable logistic regression (statsmodels): adjusted odds ratios.
    AUC is computed with scipy's rankdata (same idea as the Mann-Whitney U test).
    Returns (table, model_info).
    """
    predictors = ["BMI", "Age", "Income", "Education", "GenHlth", "PhysHlth", "MentHlth"] + BINARY_COLUMNS
    X = sm.add_constant(df[predictors].astype(float))
    y = df["Diabetes_Binary"].astype(int)
    model = sm.Logit(y, X).fit(disp=False, maxiter=100)

    conf = model.conf_int()
    table = pd.DataFrame({
        "Predictor": model.params.index,
        "Adj_Odds_Ratio": np.exp(model.params.values).round(3),
        "CI_Lower": np.exp(conf[0].values).round(3),
        "CI_Upper": np.exp(conf[1].values).round(3),
        "P_Value": model.pvalues.values,
    })
    table = table[table["Predictor"] != "const"].sort_values("Adj_Odds_Ratio", ascending=False).reset_index(drop=True)

    probabilities = model.predict(X).to_numpy()
    ranks = stats.rankdata(probabilities)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    auc = (ranks[y.to_numpy() == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    accuracy = ((probabilities >= 0.5).astype(int) == y.to_numpy()).mean()
    info = {"n": len(y), "pseudo_r2": model.prsquared, "auc": auc, "accuracy": accuracy,
            "baseline_accuracy": max(n_pos, n_neg) / len(y), "llr_p": model.llr_pvalue}
    return table, info


# ----------------------------------------------------------------------------------
# 8. RUN EVERYTHING
# ----------------------------------------------------------------------------------
def run_all_tests(df, output_folder="outputs", verbose=True):
    """Run all analyses, print them, save CSV tables and return a dictionary of results."""
    import os
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    summary_rows = []

    def section(title):
        if verbose:
            print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")

    # --- 1 descriptive ---------------------------------------------------------------
    section("1. DESCRIPTIVE STATISTICS")
    results["descriptive"] = descriptive_statistics(df)
    if verbose:
        print(results["descriptive"].to_string(index=False))

    # --- 2 assumptions ---------------------------------------------------------------
    section("2. NORMALITY AND HOMOGENEITY OF VARIANCE  (H0: normal / equal variances)")
    rows = normality_tests(df, "BMI") + normality_tests(df, "PhysHlth")
    hom_rows, levene_p = homogeneity_tests(df, "BMI")
    rows += hom_rows
    summary_rows += rows
    if verbose:
        _print_rows(rows)
        print("\n  Conclusion: p < 0.05 for normality tests -> variables are NOT normal; "
              "non-parametric tests are used as well as t-tests/ANOVA (large n makes them robust).")

    # --- 3 correlation ---------------------------------------------------------------
    section("3. CORRELATION ANALYSIS")
    rows = correlation_tests(df)
    summary_rows += rows
    results["target_correlations"] = target_correlations(df)
    if verbose:
        _print_rows(rows)
        print("\n  Spearman correlation of each variable with diabetes (ranked):")
        print(results["target_correlations"].head(10).to_string(index=False))

    # --- 4 chi-square ----------------------------------------------------------------
    section("4. CHI-SQUARE TEST OF INDEPENDENCE (factor vs diabetes)  - H0: independent")
    results["chi_square"] = chi_square_tests(df)
    if verbose:
        print(results["chi_square"].to_string(index=False))
    for _, r in results["chi_square"].head(5).iterrows():
        summary_rows.append(_row("Chi-square independence", f"{r['Factor']} vs Diabetes", r["Chi2"], r["P_Value"],
                                 f"Cramer's V = {r['Cramers_V']} ({r['Strength']})", ""))

    # --- 5 parametric ----------------------------------------------------------------
    section("5. PARAMETRIC TESTS (t-tests and ANOVA)  - H0: equal means")
    equal_var = levene_p >= ALPHA                       # choose Student or Welch based on Levene's test
    rows = (one_sample_ttest(df) + independent_ttest(df, equal_var=equal_var) + paired_ttest(df)
            + one_way_anova(df, "BMI_Capped", "Diabetes_Status"))
    anova_rows, results["anova_tables"] = anova_tables(df)
    rows += anova_rows
    rows += multi_group_anova(df, "PhysHlth", "BMI_Category")
    summary_rows += rows
    if verbose:
        print(f"  (Levene p = {format_p(levene_p)} -> {'equal' if equal_var else 'unequal'} variances -> "
              f"{'Student' if equal_var else 'Welch'} t-test chosen)")
        _print_rows(rows)
        for name, table in results["anova_tables"].items():
            print(f"\n  {name} - statsmodels anova_lm table:")
            print(table.round(4).to_string())

    # --- 6 non-parametric ------------------------------------------------------------
    section("6. NON-PARAMETRIC TESTS  - H0: same distribution")
    rows = (mann_whitney_tests(df) + kruskal_wallis_test(df, "PhysHlth", "Diabetes_Status")
            + kruskal_wallis_test(df, "BMI", "Diabetes_Status") + wilcoxon_test(df) + ks_two_sample(df))
    summary_rows += rows
    if verbose:
        _print_rows(rows)

    # --- 7 odds ratios + logistic regression -----------------------------------------
    section("7. ODDS RATIOS (unadjusted, from 2x2 tables)")
    results["odds_ratios"] = odds_ratios(df)
    if verbose:
        print(results["odds_ratios"].to_string(index=False))

    section("8. MULTIVARIABLE LOGISTIC REGRESSION (adjusted odds ratios, statsmodels)")
    table, info = logistic_regression(df)
    results["logistic"], results["logistic_info"] = table, info
    if verbose:
        print(table.to_string(index=False))
        print(f"\n  n = {info['n']:,} | Pseudo R2 = {info['pseudo_r2']:.3f} | AUC = {info['auc']:.3f} | "
              f"Accuracy = {info['accuracy']:.3f} (always predicting 'no diabetes' gives {info['baseline_accuracy']:.3f})")

    section("9. OLS REGRESSION (statsmodels) - what explains BMI?")
    ols_table, ols_info = ols_regression(df)
    results["ols"], results["ols_info"] = ols_table, ols_info
    if verbose:
        print(ols_table.to_string(index=False))
        print(f"\n  n = {ols_info['n']:,} | R2 = {ols_info['r2']:.3f} | adj R2 = {ols_info['adj_r2']:.3f} | "
              f"F-test p {format_p(ols_info['f_p'])}")

    # --- save -------------------------------------------------------------------------
    section("10. MULTIPLE-TESTING CORRECTION (Holm and Benjamini-Hochberg, statsmodels)")
    results["chi_square"] = add_multiple_testing_columns(results["chi_square"])
    results["summary"] = add_multiple_testing_columns(pd.DataFrame(summary_rows))
    n_tests = int(results["summary"]["P_Value"].notna().sum())
    raw_sig = int((results["summary"]["P_Value"] < ALPHA).sum())
    holm_sig = int((results["summary"]["Sig_After_Holm"] == "yes").sum())
    results["multiple_testing"] = {"n_tests": n_tests, "raw_significant": raw_sig, "holm_significant": holm_sig}
    if verbose:
        print(f"  {n_tests} p-values corrected together: {raw_sig} significant before correction, "
              f"{holm_sig} still significant after Holm correction.")
        not_kept = results["summary"][(results["summary"]["P_Value"] < ALPHA) & (results["summary"]["Sig_After_Holm"] == "no")]
        for _, r in not_kept.iterrows():
            print(f"    lost significance: {r['Test']} ({r['Variables']})  p = {format_p(r['P_Value'])} -> Holm {format_p(r['P_Holm'])}")
    results["summary"].to_csv(os.path.join(output_folder, "statistical_results.csv"), index=False)
    results["chi_square"].to_csv(os.path.join(output_folder, "chi_square_results.csv"), index=False)
    results["odds_ratios"].to_csv(os.path.join(output_folder, "odds_ratios.csv"), index=False)
    table.to_csv(os.path.join(output_folder, "logistic_regression.csv"), index=False)
    ols_table.to_csv(os.path.join(output_folder, "ols_regression.csv"), index=False)
    if verbose:
        print(f"\n[SAVED] statistical_results.csv, chi_square_results.csv, odds_ratios.csv, "
              f"logistic_regression.csv, ols_regression.csv -> '{output_folder}'")
    return results

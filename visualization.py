"""
visualization.py
================
All charts of the project (USER-DEFINED MODULE).

Syllabus Unit 4.1 (Matplotlib): plot(), pie, bar, histogram, area (fill_between), subplots
Syllabus Unit 4.2 (Seaborn)   : box, violin, point, line, count, bar, catplot (factor plot),
                                FacetGrid, regplot, jointplot (hexbin), pairplot, heatmap
NumPy (Unit 3.1) is used for bins (arange), bar positions (arange) and the heat-map mask (triu).

Every function  ->  builds one figure, saves it to outputs/figures/ and returns the file path.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from database_manipulation import (
    AGE_ORDER, INCOME_ORDER, EDUCATION_ORDER, HEALTH_ORDER, STATUS_ORDER, BMI_ORDER,
    RISK_FACTOR_LABELS, prevalence_by,
)

FIGURE_FOLDER = os.path.join("outputs", "figures")
STATUS_COLORS = {"No Diabetes": "#4C9BE8", "Prediabetes": "#F5A623", "Diabetes": "#D64045"}
MAIN_COLOR = "#2E86AB"
SEX_COLORS = {"Female": "#E4572E", "Male": "#2E86AB"}

sns.set_theme(style="whitegrid", font_scale=1.0)


# ----------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------
def _status_order(df):
    """Only the diabetes statuses that really exist in the data, in logical order."""
    present = df["Diabetes_Status"].unique()
    return [status for status in STATUS_ORDER if status in present]


def _save(figure, filename, show):
    """Save the figure in outputs/figures, optionally show it, then close it."""
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    path = os.path.join(FIGURE_FOLDER, filename)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return path


def _grid_figure(grid):
    """Seaborn figure-level objects store their figure in .figure (new) or .fig (old)."""
    return grid.figure if hasattr(grid, "figure") else grid.fig


# ----------------------------------------------------------------------------------
# MATPLOTLIB charts  (Unit 4.1)
# ----------------------------------------------------------------------------------
def pie_status_distribution(df, show=False):
    """Pie chart: share of respondents in every diabetes status."""
    counts = df["Diabetes_Status"].value_counts().reindex(_status_order(df))
    total = counts.sum()
    colors = [STATUS_COLORS[name] for name in counts.index]
    explode = [0.0] + [0.07] * (len(counts) - 1)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    wedges, _ = ax.pie(counts.values, colors=colors, explode=explode, startangle=90,
                       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    legend_labels = [f"{name}: {value:,} ({value / total * 100:.1f}%)" for name, value in counts.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_title("Diabetes Status of Respondents (balanced 50/50 sample)", fontsize=14, fontweight="bold")
    return _save(fig, "01_pie_diabetes_status.png", show)


def bar_prevalence_by_age(df, show=False):
    """Bar chart: diabetes (or prediabetes) rate in every age group."""
    table = prevalence_by(df, "Age_Group", AGE_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(table.index, table["Rate_Percent"], color=MAIN_COLOR, edgecolor="black")
    for bar, value in zip(bars, table["Rate_Percent"]):           # write the value above each bar
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate Rises Sharply with Age", fontsize=14, fontweight="bold")
    return _save(fig, "02_bar_diabetes_by_age.png", show)


def histogram_bmi(df, show=False):
    """Histogram: BMI distribution for every diabetes status (density so groups are comparable)."""
    bins = np.arange(12, 62, 2)                                   # NumPy arange for the bin edges
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for status in _status_order(df):
        values = df.loc[df["Diabetes_Status"] == status, "BMI"]
        ax.hist(values, bins=bins, density=True, alpha=0.5, label=f"{status} (median {values.median():.0f})",
                color=STATUS_COLORS[status], edgecolor="white")
        ax.axvline(values.median(), color=STATUS_COLORS[status], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Body Mass Index (BMI)")
    ax.set_ylabel("Density")
    ax.set_title("BMI Distribution by Diabetes Status", fontsize=14, fontweight="bold")
    ax.legend()
    return _save(fig, "03_histogram_bmi.png", show)


def grouped_bar_risk_factors(df, show=False):
    """Grouped bar chart: diabetes rate with vs without each risk factor (NumPy arange positions)."""
    columns = ["HighBP", "HighChol", "DiffWalk", "HeartDiseaseorAttack", "Stroke", "Smoker",
               "HvyAlcoholConsump", "PhysActivity", "Fruits", "Veggies"]
    labels, yes_rates, no_rates = [], [], []
    for column in columns:
        labels.append(RISK_FACTOR_LABELS[column])
        yes_rates.append(df.loc[df[column] == 1, "Diabetes_Binary"].mean() * 100)
        no_rates.append(df.loc[df[column] == 0, "Diabetes_Binary"].mean() * 100)

    positions = np.arange(len(columns))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(positions - width / 2, yes_rates, width, label="Factor present (Yes)", color="#D64045")
    ax.bar(positions + width / 2, no_rates, width, label="Factor absent (No)", color="#4C9BE8")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate With vs Without Each Health / Lifestyle Factor", fontsize=14, fontweight="bold")
    ax.legend()
    return _save(fig, "04_grouped_bar_risk_factors.png", show)


def line_area_risk_count(df, show=False):
    """plot() line with markers + shaded area (fill_between): rate vs number of risk factors."""
    table = df.groupby("Risk_Factor_Count")["Diabetes_Binary"].agg(["mean", "count"])
    table = table[table["count"] >= 100]                           # ignore very small groups
    x = table.index.to_numpy()
    y = table["mean"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, marker="o", color="#D64045", linewidth=2.5, markersize=8)
    ax.fill_between(x, y, color="#D64045", alpha=0.2)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xlabel("Number of risk factors present (0-8)\n(HighBP, HighChol, Smoker, Stroke, Heart disease, Difficulty walking, Obese, Inactive)", fontsize=9)
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate Increases with Every Added Risk Factor", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    return _save(fig, "05_line_area_risk_count.png", show)


def dashboard_subplots(df, show=False):
    """Four charts on one figure using plt.subplots(2, 2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    order = _status_order(df)

    # (0,0) pie
    counts = df["Diabetes_Status"].value_counts().reindex(order)
    axes[0, 0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, pctdistance=0.7,
                   colors=[STATUS_COLORS[s] for s in counts.index], textprops={"fontsize": 9})
    axes[0, 0].set_title("Diabetes Status")

    # (0,1) bar by age
    age_table = prevalence_by(df, "Age_Group", AGE_ORDER)
    axes[0, 1].bar(age_table.index, age_table["Rate_Percent"], color=MAIN_COLOR)
    axes[0, 1].set_title("Diabetes Rate by Age Group (%)")
    axes[0, 1].tick_params(axis="x", rotation=60)

    # (1,0) histogram
    for status in order:
        axes[1, 0].hist(df.loc[df["Diabetes_Status"] == status, "BMI"], bins=np.arange(12, 62, 2),
                        density=True, alpha=0.5, color=STATUS_COLORS[status], label=status)
    axes[1, 0].set_title("BMI Distribution")
    axes[1, 0].set_xlabel("BMI")
    axes[1, 0].legend(fontsize=8)

    # (1,1) rate by income
    income_table = prevalence_by(df, "Income_Group", INCOME_ORDER)
    axes[1, 1].plot(income_table.index, income_table["Rate_Percent"], marker="s", color="#D64045")
    axes[1, 1].set_title("Diabetes Rate by Income (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.suptitle("Diabetes Health Indicators - Overview Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_dashboard_subplots.png", show)


# ----------------------------------------------------------------------------------
# SEABORN charts - categorical variables  (Unit 4.2)
# ----------------------------------------------------------------------------------
def box_violin_bmi(df, show=False):
    """Box plot and violin plot of BMI for each diabetes status, side by side."""
    order = _status_order(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.boxplot(data=df, x="Diabetes_Status", y="BMI", order=order, hue="Diabetes_Status",
                palette=STATUS_COLORS, legend=False, ax=axes[0], fliersize=2)
    axes[0].set_title("Box Plot of BMI")
    sns.violinplot(data=df, x="Diabetes_Status", y="BMI", order=order, hue="Diabetes_Status",
                   palette=STATUS_COLORS, legend=False, ax=axes[1], inner="quartile", cut=0)
    axes[1].set_title("Violin Plot of BMI")
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle("BMI is Higher in People with Diabetes", fontsize=14, fontweight="bold")
    return _save(fig, "07_box_violin_bmi.png", show)


def count_bmi_category(df, show=False):
    """Count plot: number of respondents in every BMI category split by diabetes status."""
    order = _status_order(df)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.countplot(data=df, x="BMI_Category", order=BMI_ORDER, hue="Diabetes_Status", hue_order=order,
                  palette=STATUS_COLORS, ax=ax)
    ax.set_xlabel("BMI category (WHO)")
    ax.set_ylabel("Number of respondents")
    ax.set_title("Respondents per BMI Category and Diabetes Status", fontsize=14, fontweight="bold")
    return _save(fig, "08_count_bmi_category.png", show)


def bar_income_education(df, show=False):
    """Seaborn bar plots: diabetes rate by income and by education (pre-aggregated with groupby)."""
    income = prevalence_by(df, "Income_Group", INCOME_ORDER).reset_index()
    education = prevalence_by(df, "Education_Level", EDUCATION_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(data=income, x="Income_Group", y="Rate_Percent", order=INCOME_ORDER, color=MAIN_COLOR, ax=axes[0])
    axes[0].set_title("By Household Income")
    sns.barplot(data=education, x="Education_Level", y="Rate_Percent", order=EDUCATION_ORDER, color="#2A9D8F", ax=axes[1])
    axes[1].set_title("By Education Level")
    for ax in axes:
        ax.set_ylabel("Share with diabetes in sample (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Socio-economic Gradient: Lower Income / Education -> Higher Diabetes Rate",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "09_bar_income_education.png", show)


def point_general_health_sex(df, show=False):
    """Point plot: diabetes rate vs self-rated general health, one line per sex."""
    agg = (df.groupby(["General_Health", "Sex_Label"])["Diabetes_Binary"].mean() * 100)
    agg = agg.reset_index(name="Rate_Percent")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.pointplot(data=agg, x="General_Health", y="Rate_Percent", hue="Sex_Label", order=HEALTH_ORDER,
                  palette=SEX_COLORS, markers=["o", "s"], linestyles=["-", "--"], ax=ax)
    ax.set_xlabel("Self-rated general health")
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate by General Health and Sex", fontsize=14, fontweight="bold")
    ax.legend(title="Sex")
    return _save(fig, "10_point_health_sex.png", show)


def line_age_sex(df, show=False):
    """Seaborn line plot: diabetes rate across age codes for males and females."""
    agg = (df.groupby(["Age", "Sex_Label"])["Diabetes_Binary"].mean() * 100).reset_index(name="Rate_Percent")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=agg, x="Age", y="Rate_Percent", hue="Sex_Label", palette=SEX_COLORS, marker="o", ax=ax)
    ax.set_xticks(range(1, 14))
    ax.set_xticklabels(AGE_ORDER, rotation=45)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate Across Age: Males vs Females", fontsize=14, fontweight="bold")
    return _save(fig, "11_line_age_sex.png", show)


def catplot_age_by_sex(df, show=False):
    """Factor plot (catplot): one panel per sex showing rate in each age group."""
    agg = (df.groupby(["Age_Group", "Sex_Label"])["Diabetes_Binary"].mean() * 100).reset_index(name="Rate_Percent")
    grid = sns.catplot(data=agg, x="Age_Group", y="Rate_Percent", col="Sex_Label", kind="bar",
                       order=AGE_ORDER, color=MAIN_COLOR, height=5, aspect=1.3)
    grid.set_axis_labels("Age group", "Share with diabetes in sample (%)")
    grid.set_titles("{col_name}")
    grid.set_xticklabels(rotation=60)
    _grid_figure(grid).suptitle("Diabetes Rate by Age Group (catplot, one panel per sex)",
                                fontsize=14, fontweight="bold", y=1.03)
    return _save(_grid_figure(grid), "12_catplot_age_sex.png", show)


def facetgrid_bmi(df, show=False):
    """FacetGrid: one BMI histogram per diabetes status."""
    order = _status_order(df)
    grid = sns.FacetGrid(df, col="Diabetes_Status", col_order=order, height=4, aspect=1.1, sharey=False)
    grid.map_dataframe(sns.histplot, x="BMI", bins=np.arange(12, 62, 2), stat="percent", color=MAIN_COLOR)
    grid.set(xlim=(12, 60))
    grid.set_titles("{col_name}")
    grid.set_axis_labels("BMI", "Percent of group")
    _grid_figure(grid).suptitle("BMI Distribution in Each Diabetes Group (FacetGrid)",
                                fontsize=14, fontweight="bold", y=1.04)
    return _save(_grid_figure(grid), "13_facetgrid_bmi.png", show)


# ----------------------------------------------------------------------------------
# SEABORN charts - continuous variables  (Unit 4.2)
# ----------------------------------------------------------------------------------
def regplot_bmi_rate(df, show=False):
    """Regression plot: diabetes rate for each whole-number BMI value (quadratic fit)."""
    rounded = df.assign(BMI_Rounded=df["BMI"].round())
    agg = rounded.groupby("BMI_Rounded")["Diabetes_Binary"].agg(["mean", "count"]).reset_index()
    agg = agg[(agg["count"] >= 200) & (agg["BMI_Rounded"].between(15, 50))]
    agg["Rate_Percent"] = agg["mean"] * 100
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.regplot(data=agg, x="BMI_Rounded", y="Rate_Percent", order=2, ax=ax,
                scatter_kws={"s": 45, "color": "#D64045"}, line_kws={"color": "black"})
    ax.set_xlabel("BMI (rounded)")
    ax.set_ylabel("Share with diabetes in sample (%)")
    ax.set_title("Diabetes Rate Climbs Steadily with BMI (regression fit)", fontsize=14, fontweight="bold")
    return _save(fig, "14_regplot_bmi_rate.png", show)


def jointplot_hexbin(df, show=False):
    """Joint hexbin plot: BMI vs number of physically unhealthy days (with marginal histograms)."""
    subset = df[df["BMI"] <= 60]
    grid = sns.jointplot(data=subset, x="BMI", y="PhysHlth", kind="hex", gridsize=30,
                         color="#D64045", height=7, marginal_kws={"bins": 30},
                         joint_kws={"bins": "log", "mincnt": 1})
    grid.set_axis_labels("BMI", "Days of poor physical health (last 30)")
    _grid_figure(grid).suptitle("Joint Hexbin Plot: BMI vs Poor Physical Health Days",
                                fontsize=13, fontweight="bold", y=1.02)
    return _save(_grid_figure(grid), "15_jointplot_hexbin.png", show)


def pairplot_sample(df, show=False):
    """Pair plot of key continuous variables for a random sample of 1500 respondents."""
    sample = df.sample(n=min(1500, len(df)), random_state=42)
    grid = sns.pairplot(sample, vars=["BMI", "PhysHlth", "MentHlth", "Age"], hue="Diabetes_Status",
                        hue_order=_status_order(df), palette=STATUS_COLORS, diag_kind="hist",
                        plot_kws={"alpha": 0.5, "s": 14}, height=2.3)
    _grid_figure(grid).suptitle("Pair Plot (random sample of 1,500 respondents)", fontsize=14,
                                fontweight="bold", y=1.02)
    return _save(_grid_figure(grid), "16_pairplot_sample.png", show)


def heatmap_correlation(df, show=False):
    """Heat map of the correlation matrix (upper triangle hidden with np.triu)."""
    columns = ["Diabetes_Binary", "BMI", "Age", "Income", "Education", "GenHlth", "PhysHlth", "MentHlth",
               "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "DiffWalk", "Smoker",
               "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "Sex", "Risk_Factor_Count"]
    corr = df[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heat Map of Health Indicators", fontsize=14, fontweight="bold")
    return _save(fig, "17_heatmap_correlation.png", show)


def bar_target_correlation(df, show=False):
    """Horizontal bars: which variables are most correlated with diabetes."""
    columns = ["BMI", "Age", "Income", "Education", "GenHlth", "PhysHlth", "MentHlth", "HighBP", "HighChol",
               "HeartDiseaseorAttack", "Stroke", "DiffWalk", "Smoker", "PhysActivity", "Fruits", "Veggies",
               "HvyAlcoholConsump", "Sex", "Risk_Factor_Count"]
    corr = df[columns + ["Diabetes_Binary"]].corr()["Diabetes_Binary"].drop("Diabetes_Binary").sort_values()
    colors = ["#D64045" if value > 0 else "#4C9BE8" for value in corr.values]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with diabetes (red = raises risk, blue = lowers risk)")
    ax.set_title("Strongest Correlates of Diabetes", fontsize=14, fontweight="bold")
    return _save(fig, "18_bar_target_correlation.png", show)


# ----------------------------------------------------------------------------------
# Run all charts
# ----------------------------------------------------------------------------------
ALL_CHARTS = [
    ("Pie - diabetes status", pie_status_distribution),
    ("Bar - rate by age", bar_prevalence_by_age),
    ("Histogram - BMI", histogram_bmi),
    ("Grouped bar - risk factors", grouped_bar_risk_factors),
    ("Line + area - risk count", line_area_risk_count),
    ("Subplots dashboard", dashboard_subplots),
    ("Box + violin - BMI", box_violin_bmi),
    ("Count plot - BMI category", count_bmi_category),
    ("Bar - income & education", bar_income_education),
    ("Point plot - health x sex", point_general_health_sex),
    ("Line plot - age x sex", line_age_sex),
    ("Catplot - age by sex", catplot_age_by_sex),
    ("FacetGrid - BMI", facetgrid_bmi),
    ("Regplot - BMI vs rate", regplot_bmi_rate),
    ("Joint hexbin - BMI vs PhysHlth", jointplot_hexbin),
    ("Pair plot - sample", pairplot_sample),
    ("Heat map - correlation", heatmap_correlation),
    ("Bar - correlation with target", bar_target_correlation),
]


def generate_all_charts(df, show=False):
    """Create every chart; an error in one chart does not stop the others."""
    saved = []
    for number, (name, function) in enumerate(ALL_CHARTS, start=1):
        try:
            path = function(df, show=show)
            saved.append(path)
            print(f"  [{number:02d}/{len(ALL_CHARTS)}] {name:<34} -> {path}")
        except Exception as error:                      # keep going if one chart fails
            plt.close("all")
            print(f"  [{number:02d}/{len(ALL_CHARTS)}] {name:<34} FAILED: {error}")
    print(f"\n[DONE] {len(saved)} charts saved in '{FIGURE_FOLDER}'")
    return saved

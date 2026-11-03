"""
main.py
=======
Analysis of Factors Associated with Diabetes Status
Capstone Mini Project - Python for Data Science (S.Y. B.Tech EXCP) - Healthcare domain

Dataset : CDC BRFSS 2015 "Diabetes Health Indicators"  (Kaggle - Alex Teboul), survey responses (our file: 70,692 rows, balanced 50/50 sample)
Project files
    main.py                   -> menu-driven application (this file)
    database_manipulation.py  -> loading, cleaning, feature engineering, group-by summaries
    visualization.py          -> Matplotlib / Seaborn charts
    statistical_analysis.py   -> SciPy hypothesis tests + Statsmodels (Tukey, ANOVA/OLS, Holm/FDR, logistic)
    outputs/cleaned_diabetes_data.csv  -> cleaned data produced by this program

Run :  python main.py            (interactive menu)
       python main.py --auto     (runs the complete pipeline without questions)
"""

import os
import sys
import time
import datetime

import database_manipulation as dm           # user-defined modules (Unit 2.2)
import visualization as viz
import statistical_analysis as sa            # needs scipy + statsmodels (exits with a clear message if missing)


# ----------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------
def print_title(text):
    """Print a section heading."""
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def ask_yes_no(question, default="n"):
    """Ask a y/n question; keeps asking until the input is valid (while loop + exception handling)."""
    while True:
        try:
            answer = input(f"{question} (y/n) [{default}]: ").strip().lower() or default
        except EOFError:                          # no keyboard input available
            return default == "y"
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        print("  Please type y or n.")


def load_raw_data(state):
    """Option 1 : find the CSV in the data folder and load it."""
    path = dm.find_dataset()
    if path is None:
        print(f"\n[ERROR] Dataset not found. Download it from Kaggle (Diabetes Health Indicators Dataset) and put the "
              f"CSV file inside the '{dm.DATA_FOLDER}' folder, then run again.")
        return False
    df = dm.load_data(path)
    if df is None:
        return False
    state["raw"] = df
    state["raw_path"] = path
    return True


def clean_and_save(state):
    """Option 2 : run the cleaning pipeline and export the cleaned CSV."""
    if state["raw"] is None and not load_raw_data(state):
        return False
    start = time.time()
    clean, log = dm.clean_dataset(state["raw"])
    print_title("DATA CLEANING LOG")
    for line in log:
        print(line)
    dm.save_clean_data(clean, log)
    state["clean"], state["log"] = clean, log
    print(f"[TIME] Cleaning finished in {time.time() - start:.1f} seconds")
    return True


def ensure_clean(state):
    """Make sure a clean DataFrame is available (uses cleaned CSV if it already exists)."""
    if state["clean"] is not None:
        return True
    existing = dm.load_clean_data()
    if existing is not None:
        print(f"[INFO] Using existing '{dm.CLEAN_FILE}' ({len(existing):,} rows)")
        state["clean"] = existing
        return True
    print("[INFO] No cleaned data yet -> cleaning the raw data first ...")
    return clean_and_save(state)


# ----------------------------------------------------------------------------------
# Menu actions
# ----------------------------------------------------------------------------------
def action_inspect(state):
    if load_raw_data(state):
        print_title("RAW DATA INSPECTION")
        dm.inspect_data(state["raw"])
        print_title("MISSING VALUE REPORT")
        print(dm.missing_value_report(state["raw"]).head(10).to_string())


def action_manipulation(state):
    if ensure_clean(state):
        print_title("DATA MANIPULATION & SUMMARIES")
        dm.print_manipulation_demo(state["clean"])


def action_visualisation(state, show=None):
    if not ensure_clean(state):
        return
    if show is None:
        show = ask_yes_no("Also display the charts on screen (they are always saved as PNG files)?", "n")
    print_title("DATA VISUALISATION")
    viz.generate_all_charts(state["clean"], show=show)


def action_statistics(state):
    if ensure_clean(state):
        print_title("STATISTICAL ANALYSIS")
        state["results"] = sa.run_all_tests(state["clean"])


def build_report(df, results, log):
    """Create a plain-text report with the key findings (values are computed from the data)."""
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    n = len(df)
    prevalence = df["Diabetes_Binary"].mean() * 100
    age_table = dm.prevalence_by(df, "Age_Group", dm.AGE_ORDER)
    income_table = dm.prevalence_by(df, "Income_Group", dm.INCOME_ORDER)
    risk_table = dm.prevalence_by(df, "Risk_Level", dm.RISK_LEVEL_ORDER)
    bmi_diab = df.loc[df["Diabetes_Binary"] == 1, "BMI"].mean()
    bmi_non = df.loc[df["Diabetes_Binary"] == 0, "BMI"].mean()
    # screening / access variables are consequences of being diagnosed, so they are not listed as risk factors
    not_risk = ["Cholesterol Check (5 yrs)", "Has Health Coverage", "Could Not Afford Doctor"]
    top_or = results["odds_ratios"][~results["odds_ratios"]["Factor"].isin(not_risk)].head(4)
    top_chi = results["chi_square"].head(3)
    protective = results["odds_ratios"][(results["odds_ratios"]["Meaning"] == "lowers risk")
                                        & (results["odds_ratios"]["Factor"] != "Heavy Alcohol Use")]
    protective = protective.sort_values("Odds_Ratio")

    lines = []
    add = lines.append
    add("=" * 90)
    add("ANALYSIS OF FACTORS ASSOCIATED WITH DIABETES STATUS - AUTO-GENERATED SUMMARY")
    add(f"Generated on {now}")
    add("=" * 90)
    add("\n1. DATASET")
    add(f"   Cleaned dataset: {n:,} respondents, {df.shape[1]} columns (CDC BRFSS 2015 survey).")
    add(f"   Share of respondents with diabetes in this file: {prevalence:.1f}%  (the file is a BALANCED sample, so this is NOT the real population prevalence)")

    add("\n2. DATA CLEANING SUMMARY")
    for line in log:
        add("   " + line)

    add("\n3. KEY DESCRIPTIVE FINDINGS")
    add(f"   - Share with diabetes rises from {age_table['Rate_Percent'].iloc[0]:.1f}% (18-24 yrs) to "
        f"{age_table['Rate_Percent'].max():.1f}% (peak age group: {age_table['Rate_Percent'].idxmax()}).")
    add(f"   - Lowest-income group: {income_table['Rate_Percent'].iloc[0]:.1f}%  vs  highest-income group: "
        f"{income_table['Rate_Percent'].iloc[-1]:.1f}%.")
    add(f"   - Mean BMI: {bmi_diab:.1f} (diabetes)  vs  {bmi_non:.1f} (no diabetes).")
    add(f"   - Risk level: Low {risk_table['Rate_Percent'].iloc[0]:.1f}% | Moderate {risk_table['Rate_Percent'].iloc[1]:.1f}% "
        f"| High {risk_table['Rate_Percent'].iloc[2]:.1f}%.")

    add("\n4. KEY STATISTICAL FINDINGS (alpha = 0.05)")
    add("   - Normality tests reject normality for BMI and PhysHlth -> non-parametric tests also used.")
    for _, r in top_chi.iterrows():
        add(f"   - Chi-square: {r['Factor']} is associated with diabetes (chi2 = {r['Chi2']:,.0f}, "
            f"p {sa.format_p(r['P_Value'])}, Cramer's V = {r['Cramers_V']} -> {r['Strength']}).")
    for _, r in top_or.iterrows():
        add(f"   - Odds ratio: {r['Factor']} OR = {r['Odds_Ratio']} (95% CI {r['CI_Lower']}-{r['CI_Upper']}).")
    for _, r in protective.head(3).iterrows():
        add(f"   - Protective: {r['Factor']} OR = {r['Odds_Ratio']} (95% CI {r['CI_Lower']}-{r['CI_Upper']}).")
    summary = results["summary"]
    t_row = summary[summary["Test"].str.startswith("Independent t-test")].iloc[0]
    add(f"   - {t_row['Test']}: {t_row['Interpretation']} ; p {sa.format_p(t_row['P_Value'])} ; {t_row['Effect_Size']}.")
    if results.get("logistic_info"):
        info = results["logistic_info"]
        add(f"   - Logistic regression: AUC = {info['auc']:.3f}, pseudo-R2 = {info['pseudo_r2']:.3f} "
            f"(simultaneous effect of all factors).")

    if results.get("ols_info"):
        ols = results["ols_info"]
        add(f"   - OLS regression (statsmodels) for BMI: R2 = {ols['r2']:.3f}, adjusted R2 = {ols['adj_r2']:.3f}.")
    if results.get("multiple_testing"):
        mt = results["multiple_testing"]
        add(f"   - Multiple-testing correction (Holm): {mt['holm_significant']} of {mt['raw_significant']} significant "
            f"results (out of {mt['n_tests']} p-values) remain significant after correction.")

    add("\n5. CONCLUSION")
    add("   Age, high blood pressure, high cholesterol, BMI, poor general health, difficulty walking and heart disease")
    add("   are strongly associated with diabetes, while physical activity, higher income and higher education are")
    add("   associated with lower rates. Risk grows steadily with the number of risk factors present.")

    add("\n6. LIMITATIONS")
    add("   - Cross-sectional survey: shows ASSOCIATION, not causation.")
    add("   - Heavy alcohol use appears 'protective' (OR < 1) and Cholesterol Check appears as a huge risk (OR > 1): both are")
    add("     artefacts (people diagnosed with diabetes are advised to cut alcohol and are screened more) - reverse causation.")
    add("   - Self-reported answers can be inaccurate; BMI categories are coarse.")
    add("   - With ~69,000 rows nearly every test is 'significant' -> effect sizes (Cramer's V, Cohen's d, OR) matter more.")
    add("   - The dataset is a 50/50 balanced subsample: percentages show differences BETWEEN groups, not real-world prevalence;")
    add("     odds ratios and tests remain valid. Accuracy baseline is 50%, so AUC and odds ratios are the key measures.")
    return "\n".join(lines)


def action_report(state):
    if not ensure_clean(state):
        return
    if state["results"] is None:
        state["results"] = sa.run_all_tests(state["clean"], verbose=False)
    log = state["log"] if state["log"] else ["(cleaning log: see outputs/cleaning_log.txt)"]
    text = build_report(state["clean"], state["results"], log)
    path = os.path.join(dm.OUTPUT_FOLDER, "analysis_report.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    print(text)
    print(f"\n[SAVED] {path}")


def run_full_pipeline(state, show=False):
    """Run every step one after the other."""
    start = time.time()
    if not clean_and_save(state):
        return
    print_title("DATA MANIPULATION & SUMMARIES")
    dm.print_manipulation_demo(state["clean"])
    print_title("DATA VISUALISATION")
    viz.generate_all_charts(state["clean"], show=show)
    print_title("STATISTICAL ANALYSIS")
    state["results"] = sa.run_all_tests(state["clean"])
    print_title("REPORT")
    action_report(state)
    print(f"\n[COMPLETE] Full pipeline finished in {time.time() - start:.1f} seconds. All files are in '{dm.OUTPUT_FOLDER}'.")


# ----------------------------------------------------------------------------------
# Main menu
# ----------------------------------------------------------------------------------
MENU = """
+----------------------------------------------------------------+
|   DIABETES HEALTH INDICATORS - DATA ANALYSIS APPLICATION        |
+----------------------------------------------------------------+
|  1. Load and inspect the raw dataset                           |
|  2. Clean the data and save the cleaned CSV                    |
|  3. Data manipulation and summary tables                       |
|  4. Data visualisation (18 charts)                             |
|  5. Statistical analysis (SciPy / Statsmodels)                 |
|  6. Generate the written summary report                        |
|  7. Run the COMPLETE pipeline (2 -> 6)                         |
|  0. Exit                                                       |
+----------------------------------------------------------------+"""


def main():
    state = {"raw": None, "raw_path": None, "clean": None, "log": None, "results": None}

    if "--auto" in sys.argv:                      # non-interactive mode
        run_full_pipeline(state, show=False)
        return

    actions = {
        "1": action_inspect,
        "2": clean_and_save,
        "3": action_manipulation,
        "4": action_visualisation,
        "5": action_statistics,
        "6": action_report,
        "7": run_full_pipeline,
    }

    while True:
        print(MENU)
        try:
            choice = input("Enter your choice (0-7): ").strip()
        except EOFError:
            break
        if choice == "0":
            print("\nThank you for using the application. Goodbye!")
            break
        elif choice in actions:
            try:
                actions[choice](state)
            except Exception as error:            # the menu never crashes
                print(f"\n[ERROR] Something went wrong: {error}")
        else:
            print("\n[!] Invalid choice - please enter a number from 0 to 7.")


if __name__ == "__main__":
    main()

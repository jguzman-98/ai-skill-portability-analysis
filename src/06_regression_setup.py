"""
06_regression_setup.py
Set up main regression specification and run preliminary analysis.

Main specification (to be finalized when AI exposure index arrives):

    DeltaEmp_o = beta_1 * AIExposure_o + beta_2 * SkillPort_o
              + beta_3 * (AIExposure_o x SkillPort_o) + X_o'gamma + epsilon_o

Where:
  DeltaEmp_o   = employment change for occupation o (pre vs post 2022)
  AIExposure_o = AI task exposure index
  SkillPort_o  = aggregate skill portability measure
  X_o          = controls (occupation size, broad category, etc.)

Key hypothesis: beta_3 > 0 (portability buffers AI displacement)

For now (without AI index): run portability-only regressions + prepare framework.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

PROC_DIR = os.path.expanduser("~/Downloads/capstone/data/processed")
OUT_DIR = os.path.expanduser("~/Downloads/capstone/output/tables")
os.makedirs(OUT_DIR, exist_ok=True)


def build_analysis_dataset():
    """Merge all data sources into analysis-ready dataset."""
    print("Building analysis dataset...")

    # Skill portability
    port = pd.read_csv(os.path.join(PROC_DIR, "aggregate_skill_portability.csv"))
    port = port[["occ2010", "aggregate_portability", "mean_pairwise_portability", "title"]].copy()

    # Employment changes
    emp = pd.read_csv(os.path.join(PROC_DIR, "cps_employment_changes.csv"))
    emp["occ"] = emp["occ"].astype(int)

    # Annual employment for controls
    ann_emp = pd.read_csv(os.path.join(PROC_DIR, "cps_annual_employment.csv"))

    # Compute pre-period size (log employment) as control
    pre_size = ann_emp[ann_emp["year"].between(2018, 2021)].groupby("occ").agg(
        avg_emp=("avg_monthly_employment", "mean")
    ).reset_index()
    pre_size["log_emp_pre"] = np.log(pre_size["avg_emp"])
    pre_size["occ"] = pre_size["occ"].astype(int)

    # Compute pre-trend (2018-2021 employment growth) as control
    emp_2018 = ann_emp[ann_emp["year"] == 2018].set_index("occ")["avg_monthly_employment"]
    emp_2021 = ann_emp[ann_emp["year"] == 2021].set_index("occ")["avg_monthly_employment"]
    pre_trend = ((emp_2021 - emp_2018) / emp_2018 * 100).reset_index()
    pre_trend.columns = ["occ", "pre_trend_pct"]
    pre_trend["occ"] = pre_trend["occ"].astype(int)

    # Merge
    df = port.merge(emp, left_on="occ2010", right_on="occ", how="inner")
    df = df.merge(pre_size[["occ", "log_emp_pre"]], on="occ", how="left")
    df = df.merge(pre_trend, on="occ", how="left")

    # Add broad occupation category (from OCC2010 first digit)
    def occ_category(code):
        if code < 500: return "Management/Business"
        elif code < 1000: return "Financial/Math/Science"
        elif code < 2000: return "Computer/Engineering"
        elif code < 3000: return "Education/Legal/Social"
        elif code < 3600: return "Healthcare Practitioners"
        elif code < 4000: return "Healthcare Support"
        elif code < 4200: return "Protective Services"
        elif code < 4700: return "Food/Personal Services"
        elif code < 5000: return "Sales"
        elif code < 6000: return "Office/Administrative"
        elif code < 6200: return "Farming/Fishing"
        elif code < 6800: return "Construction"
        elif code < 7000: return "Extraction"
        elif code < 7700: return "Installation/Maintenance"
        elif code < 9000: return "Production"
        elif code < 9800: return "Transportation"
        else: return "Military/Other"

    df["occ_category"] = df["occ2010"].apply(occ_category)

    # Standardize key variables
    df["port_std"] = (df["aggregate_portability"] - df["aggregate_portability"].mean()) / df["aggregate_portability"].std()
    df["emp_change"] = df["emp_pct_change_no_covid"]

    # Drop extreme outliers (> 3 SD in employment change)
    mean_ec = df["emp_change"].mean()
    std_ec = df["emp_change"].std()
    df["outlier"] = (df["emp_change"] - mean_ec).abs() > 3 * std_ec
    print(f"  Dropping {df['outlier'].sum()} outlier occupations")
    df = df[~df["outlier"]].copy()

    print(f"  Final analysis dataset: {len(df)} occupations")
    print(f"  Occupation categories: {df['occ_category'].nunique()}")

    return df


def run_preliminary_regressions(df):
    """Run portability-only regressions (before AI index arrives)."""
    print("\n" + "=" * 60)
    print("PRELIMINARY REGRESSIONS (without AI exposure index)")
    print("=" * 60)

    results = {}

    # (1) Bivariate: employment change ~ portability
    print("\n--- Model 1: Bivariate ---")
    m1 = smf.ols("emp_change ~ port_std", data=df).fit()
    print(m1.summary2().tables[1].to_string())
    results["m1"] = m1

    # (2) With size control
    print("\n--- Model 2: + Log Employment ---")
    m2 = smf.ols("emp_change ~ port_std + log_emp_pre", data=df).fit()
    print(m2.summary2().tables[1].to_string())
    results["m2"] = m2

    # (3) With pre-trend control
    print("\n--- Model 3: + Pre-trend ---")
    m3 = smf.ols("emp_change ~ port_std + log_emp_pre + pre_trend_pct",
                  data=df.dropna(subset=["pre_trend_pct"])).fit()
    print(m3.summary2().tables[1].to_string())
    results["m3"] = m3

    # (4) With occupation category FE
    print("\n--- Model 4: + Occupation Category FE ---")
    m4 = smf.ols("emp_change ~ port_std + log_emp_pre + pre_trend_pct + C(occ_category)",
                  data=df.dropna(subset=["pre_trend_pct"])).fit()
    # Print just key coefficients
    key_vars = ["Intercept", "port_std", "log_emp_pre", "pre_trend_pct"]
    coef_table = m4.summary2().tables[1]
    print(coef_table.loc[coef_table.index.isin(key_vars)].to_string())
    print(f"  (Occupation category FE included, {df['occ_category'].nunique()} categories)")
    print(f"  R-squared: {m4.rsquared:.4f}")
    print(f"  Observations: {m4.nobs:.0f}")
    results["m4"] = m4

    return results


def create_regression_table(results):
    """Create a formatted regression table."""
    rows = []
    for var_name, display_name in [
        ("port_std", "Skill Portability (std)"),
        ("log_emp_pre", "Log Employment (pre)"),
        ("pre_trend_pct", "Pre-trend (%)"),
    ]:
        coef_row = [display_name]
        se_row = [""]
        for key in ["m1", "m2", "m3", "m4"]:
            model = results[key]
            if var_name in model.params:
                coef = model.params[var_name]
                se = model.bse[var_name]
                pval = model.pvalues[var_name]
                stars = ""
                if pval < 0.01: stars = "***"
                elif pval < 0.05: stars = "**"
                elif pval < 0.10: stars = "*"
                coef_row.append(f"{coef:.3f}{stars}")
                se_row.append(f"({se:.3f})")
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row)
        rows.append(se_row)

    # Add R2 and N
    rows.append([""] * 5)
    r2_row = ["R-squared"]
    n_row = ["Observations"]
    fe_row = ["Occ. Category FE"]
    for key in ["m1", "m2", "m3", "m4"]:
        model = results[key]
        r2_row.append(f"{model.rsquared:.4f}")
        n_row.append(f"{model.nobs:.0f}")
    fe_row += ["No", "No", "No", "Yes"]
    rows.extend([r2_row, n_row, fe_row])

    table = pd.DataFrame(rows, columns=["", "(1)", "(2)", "(3)", "(4)"])
    return table


def show_template_specification():
    """Print the main specification to be run when AI index arrives."""
    print("\n" + "=" * 60)
    print("MAIN SPECIFICATION (template for when AI index arrives)")
    print("=" * 60)
    print("""
    DeltaEmp_o = b1*AIExposure_o + b2*SkillPort_o
               + b3*(AIExposure_o * SkillPort_o)
               + b4*log(Emp_pre) + b5*PreTrend
               + CategoryFE + epsilon

    Hypothesis: b3 > 0
    (Higher portability buffers the negative effect of AI exposure)

    Variants to run:
    (1) DeltaEmp ~ AIExposure
    (2) DeltaEmp ~ AIExposure + SkillPort
    (3) DeltaEmp ~ AIExposure * SkillPort
    (4) DeltaEmp ~ AIExposure * SkillPort + controls
    (5) DeltaEmp ~ AIExposure * SkillPort + controls + CategoryFE

    Alternative outcomes:
    - Log employment change
    - Employment-to-population ratio change
    - Wage changes (if available from CPS)

    Robustness:
    - Alternative AI exposure measures
    - Drop 2020 from pre-period (COVID)
    - Weighted by pre-period employment
    - Exclude small occupations
    - Placebo: pre-2022 period (2015-2018 vs 2018-2021)
    """)


def main():
    df = build_analysis_dataset()
    results = run_preliminary_regressions(df)

    table = create_regression_table(results)
    print("\n" + "=" * 60)
    print("REGRESSION TABLE: Skill Portability and Employment Changes")
    print("=" * 60)
    print(table.to_string(index=False))

    # Save
    table.to_csv(os.path.join(OUT_DIR, "table1_preliminary_regressions.csv"), index=False)
    df.to_csv(os.path.join(PROC_DIR, "analysis_dataset.csv"), index=False)

    show_template_specification()

    print(f"\nSaved analysis dataset ({len(df)} obs) and regression table.")


if __name__ == "__main__":
    main()

"""
02_process_cps.py
Process CPS microdata into:
  1. Occupation-to-occupation switching matrix (from OCCLY -> OCC2010)
  2. Annual employment counts by occupation
  3. Pre/post 2022 employment outcomes for analysis

Uses IPUMS CPS extract with OCC2010 (Census 2010 occ codes).
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "/Users/jacobguzman/Downloads/cps_00001.csv.gz"
OUT_DIR = os.path.expanduser("~/Downloads/capstone/data/processed")
os.makedirs(OUT_DIR, exist_ok=True)

# IPUMS CPS code mappings
# EMPSTAT: 10,12 = employed; 21,22 = unemployed; 32,34,36 = NILF
EMPLOYED_CODES = [10, 12]
UNEMPLOYED_CODES = [21, 22]


def main():
    print("Loading CPS data...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Loaded {len(df):,} rows, {df.YEAR.min()}-{df.YEAR.max()}")

    # =========================================================
    # 1. SWITCHING MATRIX (from ASEC supplement, where OCCLY exists)
    # =========================================================
    print("\n--- Building Switching Matrix ---")

    # Filter to observations with valid current and prior-year occupation
    has_switch = df["OCCLY"].notna() & (df["OCC2010"] != 9999) & (df["OCCLY"] != 9999)
    switch_df = df[has_switch].copy()

    # Also need to be employed in current period
    switch_df = switch_df[switch_df["EMPSTAT"].isin(EMPLOYED_CODES)].copy()

    # OCCLY values: need to check for NIU codes
    # IPUMS OCCLY: 0 = NIU, 9999 = missing
    switch_df = switch_df[(switch_df["OCCLY"] != 0) & (switch_df["OCC2010"] != 0)].copy()

    print(f"  Valid switching observations: {len(switch_df):,}")
    print(f"  Year range: {switch_df.YEAR.min()}-{switch_df.YEAR.max()}")

    # Identify switches (different occupation from last year)
    switch_df["is_switch"] = switch_df["OCC2010"] != switch_df["OCCLY"]
    n_switches = switch_df["is_switch"].sum()
    n_stayers = (~switch_df["is_switch"]).sum()
    print(f"  Stayers: {n_stayers:,} | Switchers: {n_switches:,}")
    print(f"  Switch rate: {100 * n_switches / len(switch_df):.1f}%")

    # Build directional switching matrix using ASEC weights
    # OCCLY is only in ASEC supplement, which uses ASECWT (not WTFINL)
    switch_only = switch_df[switch_df["is_switch"]].copy()

    switching_matrix = switch_only.groupby(
        ["OCCLY", "OCC2010"]
    )["ASECWT"].sum().reset_index()
    switching_matrix.columns = ["occ_origin", "occ_dest", "weighted_switches"]

    # Also compute stayer counts by occupation (for denominator)
    stayer_counts = switch_df[~switch_df["is_switch"]].groupby(
        "OCC2010"
    )["ASECWT"].sum().reset_index()
    stayer_counts.columns = ["occ", "weighted_stayers"]

    print(f"  Unique origin occupations: {switching_matrix['occ_origin'].nunique()}")
    print(f"  Unique destination occupations: {switching_matrix['occ_dest'].nunique()}")
    print(f"  Unique directional pairs with >0 switches: {len(switching_matrix):,}")

    # Save
    switching_matrix.to_csv(
        os.path.join(OUT_DIR, "cps_switching_matrix.csv"), index=False
    )
    stayer_counts.to_csv(
        os.path.join(OUT_DIR, "cps_stayer_counts.csv"), index=False
    )

    # =========================================================
    # 2. ANNUAL EMPLOYMENT BY OCCUPATION
    # =========================================================
    print("\n--- Building Annual Employment Counts ---")

    # Use full CPS sample (not just ASEC) for employment counts
    employed = df[
        (df["EMPSTAT"].isin(EMPLOYED_CODES)) & (df["OCC2010"] != 9999)
    ].copy()

    annual_emp = employed.groupby(
        ["YEAR", "OCC2010"]
    )["WTFINL"].sum().reset_index()
    annual_emp.columns = ["year", "occ", "weighted_employment"]

    # Normalize: divide by number of months in each year (since monthly CPS)
    months_per_year = df.groupby("YEAR")["MONTH"].nunique().to_dict()
    annual_emp["n_months"] = annual_emp["year"].map(months_per_year)
    annual_emp["avg_monthly_employment"] = (
        annual_emp["weighted_employment"] / annual_emp["n_months"]
    )

    print(f"  Occupation-year observations: {len(annual_emp):,}")
    print(f"  Unique occupations: {annual_emp['occ'].nunique()}")

    annual_emp.to_csv(
        os.path.join(OUT_DIR, "cps_annual_employment.csv"), index=False
    )

    # =========================================================
    # 3. PRE/POST 2022 EMPLOYMENT CHANGES
    # =========================================================
    print("\n--- Computing Pre/Post 2022 Employment Changes ---")

    # Pre period: 2018-2021 average (exclude 2020 COVID distortion? keep for now)
    # Post period: 2023-2025
    pre = annual_emp[annual_emp["year"].between(2018, 2021)].groupby("occ")[
        "avg_monthly_employment"
    ].mean().reset_index()
    pre.columns = ["occ", "emp_pre"]

    post = annual_emp[annual_emp["year"].between(2023, 2025)].groupby("occ")[
        "avg_monthly_employment"
    ].mean().reset_index()
    post.columns = ["occ", "emp_post"]

    changes = pre.merge(post, on="occ", how="inner")
    changes["emp_change"] = changes["emp_post"] - changes["emp_pre"]
    changes["emp_pct_change"] = (
        (changes["emp_post"] - changes["emp_pre"]) / changes["emp_pre"] * 100
    )

    # Also compute without 2020 (COVID year)
    pre_no_covid = annual_emp[
        annual_emp["year"].isin([2018, 2019, 2021])
    ].groupby("occ")["avg_monthly_employment"].mean().reset_index()
    pre_no_covid.columns = ["occ", "emp_pre_no_covid"]

    changes = changes.merge(pre_no_covid, on="occ", how="left")
    changes["emp_pct_change_no_covid"] = (
        (changes["emp_post"] - changes["emp_pre_no_covid"])
        / changes["emp_pre_no_covid"] * 100
    )

    print(f"  Occupations with pre & post data: {len(changes)}")
    print(f"\n  Employment change distribution (pct, excl COVID):")
    print(f"    Mean:   {changes['emp_pct_change_no_covid'].mean():+.1f}%")
    print(f"    Median: {changes['emp_pct_change_no_covid'].median():+.1f}%")
    print(f"    Std:    {changes['emp_pct_change_no_covid'].std():.1f}%")
    print(f"    Min:    {changes['emp_pct_change_no_covid'].min():+.1f}%")
    print(f"    Max:    {changes['emp_pct_change_no_covid'].max():+.1f}%")

    # Top decliners and growers
    print(f"\n  Top 10 declining occupations (pct change, excl COVID):")
    for _, row in changes.nsmallest(10, "emp_pct_change_no_covid").iterrows():
        print(f"    OCC {int(row['occ']):>4d}: {row['emp_pct_change_no_covid']:+.1f}%")

    print(f"\n  Top 10 growing occupations (pct change, excl COVID):")
    for _, row in changes.nlargest(10, "emp_pct_change_no_covid").iterrows():
        print(f"    OCC {int(row['occ']):>4d}: {row['emp_pct_change_no_covid']:+.1f}%")

    changes.to_csv(
        os.path.join(OUT_DIR, "cps_employment_changes.csv"), index=False
    )

    # =========================================================
    # 4. SUMMARY STATISTICS
    # =========================================================
    print("\n--- Overall Summary ---")
    total_emp_by_year = annual_emp.groupby("year")["avg_monthly_employment"].sum()
    print("  Total weighted employment by year:")
    for year, emp in total_emp_by_year.items():
        print(f"    {year}: {emp/1e6:.1f}M")

    print(f"\nDone! Files saved to {OUT_DIR}")
    print("  - cps_switching_matrix.csv")
    print("  - cps_stayer_counts.csv")
    print("  - cps_annual_employment.csv")
    print("  - cps_employment_changes.csv")


if __name__ == "__main__":
    main()

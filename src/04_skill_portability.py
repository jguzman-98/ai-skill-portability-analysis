"""
04_skill_portability.py
Replicate Khachiyan (2021) skill portability measure using O*NET + CPS data.

Two-step procedure:
  Step 1: OLS - regress log(switches/stayers) on origin + destination FE
          to get residuals (pair-specific switching unexplained by occupation FE)
  Step 2: Random Forest - predict residuals from skill vectors of both occupations
          Predicted value = "Skill Portability" measure

Output: directional pairwise skill portability matrix + aggregated occupation-level scores.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.expanduser("~/Downloads/capstone/data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
OUT_DIR = os.path.expanduser("~/Downloads/capstone/output")
os.makedirs(OUT_DIR, exist_ok=True)


def build_switching_regression_data():
    """
    Build the dataset for the Cortes & Gallipoli (2018) style regression.
    Unit of observation: directional occupation pair (origin -> destination).
    Dependent variable: ln(switches from o to o') / ln(stayers in o).
    """
    print("Building switching regression data...")

    # Load switching matrix and stayer counts
    switches = pd.read_csv(os.path.join(PROC_DIR, "cps_switching_matrix.csv"))
    stayers = pd.read_csv(os.path.join(PROC_DIR, "cps_stayer_counts.csv"))

    # Ensure integer occupation codes for matching
    switches["occ_origin"] = switches["occ_origin"].astype(int)
    switches["occ_dest"] = switches["occ_dest"].astype(int)
    stayers["occ"] = stayers["occ"].astype(int)

    # Load skill matrix
    skills = pd.read_csv(os.path.join(PROC_DIR, "skill_matrix_by_occ2010.csv"), index_col=0)
    skills.index = skills.index.astype(int)
    skill_occs = set(skills.index)

    # Filter to occupations with skill data
    switches = switches[
        switches["occ_origin"].isin(skill_occs) &
        switches["occ_dest"].isin(skill_occs)
    ].copy()
    stayers = stayers[stayers["occ"].isin(skill_occs)].copy()

    # Merge stayer counts as denominator
    switches = switches.merge(
        stayers, left_on="occ_origin", right_on="occ", how="inner"
    )

    # Compute log switching share: ln(s_{o,o'} / s_{o,o})
    # Replace zeros with small value (Khachiyan approach)
    min_nonzero = switches.loc[
        switches["weighted_switches"] > 0, "weighted_switches"
    ].min()

    switches["switch_share"] = switches["weighted_switches"] / switches["weighted_stayers"]
    switches.loc[switches["switch_share"] == 0, "switch_share"] = min_nonzero / switches["weighted_stayers"]
    switches["log_switch_share"] = np.log(switches["switch_share"])

    # Remove self-pairs and infinite values
    switches = switches[switches["occ_origin"] != switches["occ_dest"]].copy()
    switches = switches[np.isfinite(switches["log_switch_share"])].copy()

    print(f"  Directional pairs: {len(switches):,}")
    print(f"  Unique origins: {switches['occ_origin'].nunique()}")
    print(f"  Unique destinations: {switches['occ_dest'].nunique()}")
    print(f"  Log switch share range: [{switches['log_switch_share'].min():.2f}, {switches['log_switch_share'].max():.2f}]")

    return switches, skills


def step1_ols_residualize(switches):
    """
    Step 1: OLS regression of log switching shares on occupation fixed effects.
    ln(s_{o,o'}/s_{o,o}) = alpha_o + lambda_{o'} + epsilon_{o,o'}

    Uses demeaning approach for efficiency with many FE.
    """
    print("\nStep 1: OLS residualization on occupation FE...")

    y = switches["log_switch_share"].values

    # Create origin and destination fixed effects using dummies
    # For efficiency, demean by origin and destination
    origin_means = switches.groupby("occ_origin")["log_switch_share"].transform("mean")
    dest_means = switches.groupby("occ_dest")["log_switch_share"].transform("mean")
    grand_mean = y.mean()

    # Two-way FE approximation via iterative demeaning
    residuals = y.copy()
    for _ in range(50):  # Iterate to convergence
        residuals_old = residuals.copy()
        # Demean by origin
        origin_resid_means = pd.Series(residuals, index=switches.index).groupby(
            switches["occ_origin"]
        ).transform("mean")
        residuals = residuals - origin_resid_means.values
        # Demean by destination
        dest_resid_means = pd.Series(residuals, index=switches.index).groupby(
            switches["occ_dest"]
        ).transform("mean")
        residuals = residuals - dest_resid_means.values

        # Check convergence
        if np.max(np.abs(residuals - residuals_old)) < 1e-10:
            break

    switches = switches.copy()
    switches["residual"] = residuals

    # R-squared of FE model
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_resid = np.sum(residuals ** 2)
    r2_fe = 1 - ss_resid / ss_total

    print(f"  FE model R²: {r2_fe:.4f}")
    print(f"  Residual std: {np.std(residuals):.4f}")

    return switches, r2_fe


def step2_random_forest(switches, skills):
    """
    Step 2: Random Forest to predict residual switching from skill data.
    Features: skill vectors of origin, destination, and their differences.
    Target: OLS residuals from Step 1.
    Predicted value = Skill Portability measure.
    """
    print("\nStep 2: Random Forest on skill vectors...")

    # Build feature matrix
    skill_cols = skills.columns.tolist()

    # For each directional pair, create features:
    # [skills_origin, skills_dest, skills_origin - skills_dest]
    origins = switches["occ_origin"].values
    dests = switches["occ_dest"].values

    # Get skill vectors
    origin_skills = skills.loc[origins].values
    dest_skills = skills.loc[dests].values
    diff_skills = origin_skills - dest_skills

    X = np.hstack([origin_skills, dest_skills, diff_skills])
    y = switches["residual"].values

    feature_names = (
        [f"origin_{c}" for c in skill_cols] +
        [f"dest_{c}" for c in skill_cols] +
        [f"diff_{c}" for c in skill_cols]
    )

    print(f"  Feature matrix: {X.shape}")
    print(f"  Target (residuals): {len(y)}")

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    # Cross-validation score
    print("  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"  CV R² scores: {cv_scores}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full data
    print("  Fitting full model...")
    rf.fit(X, y)

    # In-sample predictions = skill portability measure
    switches = switches.copy()
    switches["predicted_skill_portability"] = rf.predict(X)

    # Training R²
    train_r2 = rf.score(X, y)
    print(f"  Training R²: {train_r2:.4f}")

    # Feature importance (top 20)
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    print("\n  Top 20 most important features:")
    for feat, imp in importances.nlargest(20).items():
        print(f"    {feat}: {imp:.4f}")

    return switches, rf, importances


def compute_aggregate_portability(switches, skills):
    """
    Compute occupation-level aggregate skill portability.
    For each occupation, weighted sum of pairwise portabilities to all other occs,
    weighted by employment in each destination.
    """
    print("\nComputing aggregate skill portability by occupation...")

    # Load employment data
    emp = pd.read_csv(os.path.join(PROC_DIR, "cps_employment_changes.csv"))
    emp_map = dict(zip(emp["occ"].astype(int), emp["emp_pre"]))

    # Normalize portability to [0, 1]
    port = switches[["occ_origin", "occ_dest", "predicted_skill_portability"]].copy()
    port["portability_norm"] = (
        (port["predicted_skill_portability"] - port["predicted_skill_portability"].min()) /
        (port["predicted_skill_portability"].max() - port["predicted_skill_portability"].min())
    )

    # For each origin occupation, compute weighted portability
    agg_port = {}
    for occ_o, group in port.groupby("occ_origin"):
        weighted_port = 0
        total_weight = 0
        for _, row in group.iterrows():
            occ_d = row["occ_dest"]
            emp_d = emp_map.get(int(occ_d), 0)
            weighted_port += row["portability_norm"] * emp_d
            total_weight += emp_d

        if total_weight > 0:
            agg_port[int(occ_o)] = weighted_port / total_weight

    agg_df = pd.DataFrame({
        "occ2010": list(agg_port.keys()),
        "aggregate_portability": list(agg_port.values())
    })

    # Also compute simpler version: mean pairwise portability
    mean_port = port.groupby("occ_origin")["portability_norm"].mean().reset_index()
    mean_port.columns = ["occ2010", "mean_pairwise_portability"]
    mean_port["occ2010"] = mean_port["occ2010"].astype(int)

    agg_df = agg_df.merge(mean_port, on="occ2010", how="left")

    # Add titles
    titles = pd.read_csv(os.path.join(PROC_DIR, "occ2010_titles.csv"))
    agg_df = agg_df.merge(titles, on="occ2010", how="left")

    # Sort by portability
    agg_df = agg_df.sort_values("aggregate_portability", ascending=False)

    print(f"  Occupations with portability scores: {len(agg_df)}")
    print(f"\n  Top 10 most portable occupations:")
    for _, row in agg_df.head(10).iterrows():
        print(f"    {row['title'][:50]:<50s} {row['aggregate_portability']:.4f}")

    print(f"\n  Bottom 10 least portable occupations:")
    for _, row in agg_df.tail(10).iterrows():
        print(f"    {row['title'][:50]:<50s} {row['aggregate_portability']:.4f}")

    return agg_df, port


def main():
    print("=" * 60)
    print("SKILL PORTABILITY MODEL (Khachiyan 2021 replication)")
    print("=" * 60)

    # Build regression data
    switches, skills = build_switching_regression_data()

    # Step 1: OLS residualization
    switches, r2_fe = step1_ols_residualize(switches)

    # Step 2: Random Forest
    switches, rf_model, importances = step2_random_forest(switches, skills)

    # Aggregate portability
    agg_df, pairwise_port = compute_aggregate_portability(switches, skills)

    # Save outputs
    print("\nSaving outputs...")

    # Pairwise portability
    pairwise_out = switches[
        ["occ_origin", "occ_dest", "log_switch_share", "residual",
         "predicted_skill_portability"]
    ].copy()
    pairwise_out.to_csv(os.path.join(PROC_DIR, "pairwise_skill_portability.csv"), index=False)

    # Aggregate portability
    agg_df.to_csv(os.path.join(PROC_DIR, "aggregate_skill_portability.csv"), index=False)

    # Feature importances
    importances.sort_values(ascending=False).to_csv(
        os.path.join(OUT_DIR, "rf_feature_importances.csv")
    )

    print(f"\nDone!")
    print(f"  Pairwise portability: {len(pairwise_out):,} directional pairs")
    print(f"  Aggregate portability: {len(agg_df)} occupations")
    print(f"  FE R²: {r2_fe:.4f}")


if __name__ == "__main__":
    main()

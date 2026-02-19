"""
01_process_onet.py
Process O*NET 30.1 data into occupation-level skill vectors.

Combines Skills, Abilities, and Knowledge data into a single
occupation x skill-dimension matrix suitable for computing
pairwise skill portability measures.

Uses the "Level" (LV) scale values, which measure the degree
to which each skill/ability/knowledge is required.
"""

import pandas as pd
import numpy as np
import os

# Paths
RAW_DIR = os.path.expanduser("~/Downloads/capstone/data/raw/db_30_1_text")
OUT_DIR = os.path.expanduser("~/Downloads/capstone/data/processed")
os.makedirs(OUT_DIR, exist_ok=True)


def load_onet_file(filename, category_label):
    """Load an O*NET data file and filter to Level (LV) scale."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, sep="\t")

    # Keep only the Level scale (LV) - measures degree of requirement
    df = df[df["Scale ID"] == "LV"].copy()

    # Drop suppressed values
    df = df[df["Recommend Suppress"] != "Y"].copy()

    # Create a clean element name with category prefix to avoid collisions
    df["dimension"] = category_label + ": " + df["Element Name"]

    # Keep only what we need
    df = df[["O*NET-SOC Code", "dimension", "Data Value"]].copy()
    df.columns = ["soc_code", "dimension", "value"]

    return df


def aggregate_to_6digit(df):
    """
    Aggregate detailed O*NET-SOC codes (e.g., 11-1011.03) to
    6-digit SOC level (e.g., 11-1011) by averaging.

    This ensures compatibility with CPS occupation codes and
    AI exposure indices which typically use 6-digit SOC.
    """
    df = df.copy()
    df["soc6"] = df["soc_code"].str[:7]  # e.g., "11-1011"

    # Average across detailed codes within same 6-digit SOC
    df_agg = df.groupby(["soc6", "dimension"])["value"].mean().reset_index()

    return df_agg


def main():
    print("Loading O*NET data files...")

    # Load Skills, Abilities, Knowledge
    skills = load_onet_file("Skills.txt", "Skill")
    abilities = load_onet_file("Abilities.txt", "Ability")
    knowledge = load_onet_file("Knowledge.txt", "Knowledge")

    print(f"  Skills: {skills.shape[0]} rows, {skills['dimension'].nunique()} dimensions")
    print(f"  Abilities: {abilities.shape[0]} rows, {abilities['dimension'].nunique()} dimensions")
    print(f"  Knowledge: {knowledge.shape[0]} rows, {knowledge['dimension'].nunique()} dimensions")

    # Combine all
    combined = pd.concat([skills, abilities, knowledge], ignore_index=True)
    print(f"\nCombined: {combined.shape[0]} rows")
    print(f"  Unique occupations (detailed): {combined['soc_code'].nunique()}")
    print(f"  Unique dimensions: {combined['dimension'].nunique()}")

    # Aggregate to 6-digit SOC
    combined_6d = aggregate_to_6digit(combined)
    print(f"\nAfter 6-digit aggregation:")
    print(f"  Unique occupations: {combined_6d['soc6'].nunique()}")

    # Pivot to occupation x dimension matrix
    skill_matrix = combined_6d.pivot_table(
        index="soc6",
        columns="dimension",
        values="value",
        aggfunc="mean"
    )

    # Handle missing values - fill with 0 (skill not relevant to occupation)
    n_missing = skill_matrix.isna().sum().sum()
    n_total = skill_matrix.shape[0] * skill_matrix.shape[1]
    print(f"\nMissing values: {n_missing}/{n_total} ({100*n_missing/n_total:.1f}%)")
    skill_matrix = skill_matrix.fillna(0)

    # Standardize each dimension to [0, 1] range for comparability
    skill_matrix_norm = (skill_matrix - skill_matrix.min()) / (skill_matrix.max() - skill_matrix.min())

    print(f"\nFinal skill matrix shape: {skill_matrix_norm.shape}")
    print(f"  {skill_matrix_norm.shape[0]} occupations x {skill_matrix_norm.shape[1]} skill dimensions")

    # Save both raw and normalized versions
    skill_matrix.to_csv(os.path.join(OUT_DIR, "onet_skill_matrix_raw.csv"))
    skill_matrix_norm.to_csv(os.path.join(OUT_DIR, "onet_skill_matrix_normalized.csv"))

    # Also save occupation list with titles
    occ_data = pd.read_csv(
        os.path.join(RAW_DIR, "Occupation Data.txt"), sep="\t"
    )
    occ_data["soc6"] = occ_data["O*NET-SOC Code"].str[:7]
    occ_titles = occ_data.drop_duplicates("soc6")[["soc6", "Title"]].copy()
    occ_titles.to_csv(os.path.join(OUT_DIR, "occupation_titles.csv"), index=False)

    # Print sample
    print("\nSample occupations and top skills:")
    for soc in skill_matrix_norm.index[:3]:
        title = occ_titles[occ_titles["soc6"] == soc]["Title"].values
        title_str = title[0] if len(title) > 0 else soc
        top_skills = skill_matrix_norm.loc[soc].nlargest(5)
        print(f"\n  {soc} - {title_str}")
        for dim, val in top_skills.items():
            print(f"    {dim}: {val:.3f}")

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Dimensions with highest average values across occupations:")
    top_dims = skill_matrix_norm.mean().nlargest(10)
    for dim, val in top_dims.items():
        print(f"  {dim}: {val:.3f}")

    print(f"\nDimensions with most variation (std) across occupations:")
    var_dims = skill_matrix_norm.std().nlargest(10)
    for dim, val in var_dims.items():
        print(f"  {dim}: {val:.3f}")

    print(f"\nDone! Files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

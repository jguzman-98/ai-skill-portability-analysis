"""
05_descriptive_figures.py
Generate descriptive figures for the capstone paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

PROC_DIR = os.path.expanduser("~/Downloads/capstone/data/processed")
FIG_DIR = os.path.expanduser("~/Downloads/capstone/output/figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (8, 5),
})


def fig1_portability_distribution():
    """Distribution of aggregate skill portability across occupations."""
    port = pd.read_csv(os.path.join(PROC_DIR, "aggregate_skill_portability.csv"))

    fig, ax = plt.subplots()
    ax.hist(port["aggregate_portability"], bins=40, color="#2196F3",
            edgecolor="white", alpha=0.85)
    ax.axvline(port["aggregate_portability"].median(), color="red",
               linestyle="--", linewidth=1.5, label=f'Median: {port["aggregate_portability"].median():.3f}')
    ax.set_xlabel("Aggregate Skill Portability")
    ax.set_ylabel("Number of Occupations")
    ax.set_title("Distribution of Occupational Skill Portability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_portability_distribution.png"))
    plt.close()
    print("  Saved fig1_portability_distribution.png")


def fig2_pairwise_distribution():
    """Distribution of pairwise skill portability (predicted measure)."""
    pw = pd.read_csv(os.path.join(PROC_DIR, "pairwise_skill_portability.csv"))

    # Normalize to [0,1]
    pw["port_norm"] = (
        (pw["predicted_skill_portability"] - pw["predicted_skill_portability"].min()) /
        (pw["predicted_skill_portability"].max() - pw["predicted_skill_portability"].min())
    )

    fig, ax = plt.subplots()
    ax.hist(pw["port_norm"], bins=80, color="#4CAF50", edgecolor="white",
            alpha=0.85, density=True)
    ax.set_xlabel("Pairwise Skill Portability (Normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Pairwise Skill Portability\nAcross Directional Occupation Pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_pairwise_distribution.png"))
    plt.close()
    print("  Saved fig2_pairwise_distribution.png")


def fig3_top_bottom_occupations():
    """Bar chart of top/bottom 15 occupations by portability."""
    port = pd.read_csv(os.path.join(PROC_DIR, "aggregate_skill_portability.csv"))
    port["title_short"] = port["title"].str[:45]

    top = port.nlargest(15, "aggregate_portability")
    bottom = port.nsmallest(15, "aggregate_portability")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Top 15
    ax1.barh(range(len(top)), top["aggregate_portability"].values,
             color="#2196F3", edgecolor="white")
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top["title_short"].values, fontsize=9)
    ax1.set_xlabel("Aggregate Portability")
    ax1.set_title("Most Portable Occupations")
    ax1.invert_yaxis()

    # Bottom 15
    ax2.barh(range(len(bottom)), bottom["aggregate_portability"].values,
             color="#F44336", edgecolor="white")
    ax2.set_yticks(range(len(bottom)))
    ax2.set_yticklabels(bottom["title_short"].values, fontsize=9)
    ax2.set_xlabel("Aggregate Portability")
    ax2.set_title("Least Portable Occupations")
    ax2.invert_yaxis()

    fig.suptitle("Occupations by Skill Portability", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_top_bottom_occupations.png"),
                bbox_inches="tight")
    plt.close()
    print("  Saved fig3_top_bottom_occupations.png")


def fig4_employment_changes():
    """Pre/post 2022 employment changes distribution."""
    emp = pd.read_csv(os.path.join(PROC_DIR, "cps_employment_changes.csv"))

    fig, ax = plt.subplots()
    emp_clean = emp["emp_pct_change_no_covid"].dropna()
    emp_clean = emp_clean[emp_clean.between(-60, 100)]  # Trim outliers

    ax.hist(emp_clean, bins=50, color="#FF9800", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(emp_clean.median(), color="red", linestyle="--", linewidth=1.5,
               label=f'Median: {emp_clean.median():+.1f}%')
    ax.set_xlabel("Employment Change (%)")
    ax.set_ylabel("Number of Occupations")
    ax.set_title("Employment Changes by Occupation\nPre (2018-19, 2021 avg) vs Post (2023-25 avg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_employment_changes.png"))
    plt.close()
    print("  Saved fig4_employment_changes.png")


def fig5_portability_vs_employment():
    """Scatter: skill portability vs employment change (preview of main result)."""
    port = pd.read_csv(os.path.join(PROC_DIR, "aggregate_skill_portability.csv"))
    emp = pd.read_csv(os.path.join(PROC_DIR, "cps_employment_changes.csv"))
    emp["occ"] = emp["occ"].astype(int)

    merged = port.merge(emp, left_on="occ2010", right_on="occ", how="inner")
    merged = merged[merged["emp_pct_change_no_covid"].between(-60, 100)]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        merged["aggregate_portability"],
        merged["emp_pct_change_no_covid"],
        s=np.sqrt(merged["emp_pre"]) / 50,
        alpha=0.4, color="#673AB7", edgecolors="white", linewidth=0.3
    )

    # Add trend line
    z = np.polyfit(merged["aggregate_portability"],
                   merged["emp_pct_change_no_covid"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(merged["aggregate_portability"].min(),
                          merged["aggregate_portability"].max(), 100)
    ax.plot(x_range, p(x_range), "r--", linewidth=2, alpha=0.7)

    corr = merged["aggregate_portability"].corr(merged["emp_pct_change_no_covid"])
    ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Aggregate Skill Portability")
    ax.set_ylabel("Employment Change (%)\n(Pre vs Post 2022)")
    ax.set_title("Skill Portability vs. Employment Change\n(Bubble size = pre-period employment)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_portability_vs_employment.png"))
    plt.close()
    print("  Saved fig5_portability_vs_employment.png")


def fig6_feature_importances():
    """Top RF feature importances."""
    imp = pd.read_csv(
        os.path.expanduser("~/Downloads/capstone/output/rf_feature_importances.csv"),
        index_col=0, header=None
    )
    imp.columns = ["importance"]
    imp = imp.nlargest(25, "importance")

    # Clean names
    imp.index = imp.index.str.replace("diff_", "").str.replace("Skill: ", "").str.replace("Ability: ", "").str.replace("Knowledge: ", "")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(imp)), imp["importance"].values, color="#009688",
            edgecolor="white")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=9)
    ax.set_xlabel("Feature Importance (Increase in Node Purity)")
    ax.set_title("Top 25 Skill Dimensions Driving Portability\n(Random Forest Feature Importance)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_feature_importances.png"))
    plt.close()
    print("  Saved fig6_feature_importances.png")


def main():
    print("Generating descriptive figures...\n")
    fig1_portability_distribution()
    fig2_pairwise_distribution()
    fig3_top_bottom_occupations()
    fig4_employment_changes()
    fig5_portability_vs_employment()
    fig6_feature_importances()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

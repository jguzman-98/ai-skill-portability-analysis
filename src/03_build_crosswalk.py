"""
03_build_crosswalk.py (v2)
Build comprehensive crosswalk: CPS OCC2010 -> O*NET SOC codes.

Strategy:
  1. Use Census 2010->2018 crosswalk as primary source
  2. Use IPUMS ACS OCC->OCCSOC crosswalk as supplement
  3. Fill remaining gaps with title-based matching
  4. Average O*NET skill vectors where one CPS code -> multiple SOC codes
"""

import pandas as pd
import numpy as np
import os
from difflib import SequenceMatcher

DATA_DIR = os.path.expanduser("~/Downloads/capstone/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")


def format_soc(code):
    """Convert '111011' or '1110XX' to '11-1011' or '11-10XX' format."""
    code = str(code).strip().replace("-", "").replace(" ", "")
    if len(code) >= 6:
        return f"{code[:2]}-{code[2:6]}"
    return code


def find_onet_matches(soc_str, onet_socs):
    """Find O*NET codes matching a SOC code (handling XX wildcards)."""
    soc_str = str(soc_str).strip()

    # Direct match
    if soc_str in onet_socs:
        return [soc_str]

    # Wildcard match: XX at end means broad group
    if "X" in soc_str.upper():
        prefix = soc_str.upper().split("X")[0]
        if len(prefix) >= 4:
            matches = [s for s in onet_socs if s.replace("-", "").startswith(prefix.replace("-", ""))]
            if matches:
                return matches

    # Prefix match (first 5 chars of SOC)
    prefix5 = soc_str.replace("-", "")[:5]
    matches = [s for s in onet_socs if s.replace("-", "").startswith(prefix5)]
    if matches:
        return matches

    return []


def title_similarity(title1, title2):
    """Compute similarity between two occupation titles."""
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()
    return SequenceMatcher(None, t1, t2).ratio()


def main():
    print("=" * 60)
    print("Building comprehensive CPS OCC2010 -> O*NET crosswalk")
    print("=" * 60)

    # Load O*NET data
    onet = pd.read_csv(os.path.join(PROC_DIR, "onet_skill_matrix_normalized.csv"), index_col=0)
    onet_socs = set(onet.index)
    onet_titles = pd.read_csv(os.path.join(PROC_DIR, "occupation_titles.csv"))
    onet_title_map = dict(zip(onet_titles["soc6"], onet_titles["Title"]))

    # Load CPS employment data to know which OCC2010 codes we need
    cps_emp = pd.read_csv(os.path.join(PROC_DIR, "cps_employment_changes.csv"))
    cps_occs = set(cps_emp["occ"].astype(int).unique())

    # Load OCC2010 titles from IPUMS crosswalk
    occ2010_xwalk = pd.read_excel(os.path.join(RAW_DIR, "cps_occ2010_xwalk.xlsx"))
    occ2010_title_map = dict(zip(
        occ2010_xwalk["OCC2010 (fixed in 2025)"].astype(int),
        occ2010_xwalk["OCC2010 Title"]
    ))

    # ---- Source 1: Census 2010->2018 crosswalk ----
    print("\n[1] Census 2010->2018 crosswalk...")
    census_xw = pd.read_excel(
        os.path.join(RAW_DIR, "census_2018_occ_crosswalk.xlsx"),
        sheet_name="2010 to 2018 Crosswalk ",
        header=None, skiprows=3
    )
    census_xw.columns = ["soc_2010", "census_2010", "title_2010",
                          "soc_2018", "census_2018", "title_2018"]
    census_xw["soc_2010"] = census_xw["soc_2010"].ffill()
    census_xw["census_2010"] = census_xw["census_2010"].ffill()
    census_xw = census_xw.dropna(subset=["soc_2018"])
    census_xw["census_2010"] = pd.to_numeric(census_xw["census_2010"], errors="coerce")
    census_xw = census_xw[census_xw["census_2010"].notna()]
    census_xw["census_2010"] = census_xw["census_2010"].astype(int)
    census_xw["soc_2018_fmt"] = census_xw["soc_2018"].astype(str).str.strip()

    # Map census codes to O*NET
    mapping = {}  # occ2010 -> set of onet SOC codes

    for _, row in census_xw.iterrows():
        occ = row["census_2010"]
        soc = row["soc_2018_fmt"]
        matches = find_onet_matches(soc, onet_socs)
        if matches:
            if occ not in mapping:
                mapping[occ] = set()
            mapping[occ].update(matches)

    matched_1 = cps_occs & set(mapping.keys())
    print(f"  Matched {len(matched_1)}/{len(cps_occs)} CPS occupations")

    # ---- Source 2: IPUMS ACS crosswalk ----
    print("\n[2] IPUMS ACS OCC->OCCSOC crosswalk...")
    ipums_xw = pd.read_csv(
        os.path.join(RAW_DIR, "ipums_occ_occsoc_crosswalk.csv"),
        encoding="latin-1"
    )

    # Try multiple vintage columns
    vintage_pairs = [
        ("2010-2012 ACS/PRCS OCC code", "2010-2012 ACS/PRCS OCCSOC"),
        ("2013-2017 ACS/PRCS OCC code", "2013-2017 ACS/PRCS OCCSOC code"),
        ("2018 ACS/PRCS OCC code", "2018 Onward ACS/PRCS"),
    ]

    for occ_col, soc_col in vintage_pairs:
        sub = ipums_xw[[occ_col, soc_col]].dropna()
        sub.columns = ["occ", "occsoc"]
        sub["occ"] = pd.to_numeric(sub["occ"], errors="coerce")
        sub = sub[sub["occ"].notna() & (sub["occ"] > 0)]
        sub["occ"] = sub["occ"].astype(int)

        for _, row in sub.iterrows():
            occ = row["occ"]
            soc = format_soc(row["occsoc"])
            matches = find_onet_matches(soc, onet_socs)
            if matches and occ not in mapping:
                mapping[occ] = set()
            if matches:
                mapping[occ].update(matches)

    matched_2 = cps_occs & set(mapping.keys())
    print(f"  Matched {len(matched_2)}/{len(cps_occs)} CPS occupations (cumulative)")

    # ---- Source 3: Title-based matching for remaining ----
    print("\n[3] Title-based matching for remaining...")
    still_unmatched = cps_occs - set(mapping.keys())

    onet_title_list = [(soc, title) for soc, title in onet_title_map.items()]

    for occ in still_unmatched:
        if occ not in occ2010_title_map:
            continue
        cps_title = occ2010_title_map[occ]

        # Find best matching O*NET title
        best_score = 0
        best_matches = []
        for soc, onet_title in onet_title_list:
            score = title_similarity(cps_title, onet_title)
            if score > best_score:
                best_score = score
                best_matches = [soc]
            elif score == best_score:
                best_matches.append(soc)

        if best_score >= 0.6:  # Threshold for acceptable match
            mapping[occ] = set(best_matches)

    matched_3 = cps_occs & set(mapping.keys())
    print(f"  Matched {len(matched_3)}/{len(cps_occs)} CPS occupations (cumulative)")

    # Show remaining unmatched
    final_unmatched = cps_occs - set(mapping.keys())
    if final_unmatched:
        print(f"\n  Still unmatched ({len(final_unmatched)}):")
        for occ in sorted(final_unmatched)[:20]:
            title = occ2010_title_map.get(occ, "Unknown")
            print(f"    OCC {occ}: {title}")

    # ---- Build final skill matrix ----
    print("\n[4] Building skill matrix by OCC2010...")

    occ2010_skills = {}
    occ2010_titles = {}

    for occ, soc_set in mapping.items():
        soc_list = list(soc_set)
        matching = onet.loc[onet.index.isin(soc_list)]
        if len(matching) > 0:
            occ2010_skills[occ] = matching.mean()
            occ2010_titles[occ] = occ2010_title_map.get(occ, f"OCC {occ}")

    skill_matrix = pd.DataFrame(occ2010_skills).T
    skill_matrix.index.name = "occ2010"
    skill_matrix.index = skill_matrix.index.astype(int)

    # Final coverage check
    matched_final = cps_occs & set(skill_matrix.index)
    cps_emp_matched = cps_emp[cps_emp["occ"].astype(int).isin(skill_matrix.index)]
    emp_coverage = cps_emp_matched["emp_pre"].sum() / cps_emp["emp_pre"].sum()

    print(f"\n  Final skill matrix: {skill_matrix.shape}")
    print(f"  CPS occupation coverage: {len(matched_final)}/{len(cps_occs)} ({100*len(matched_final)/len(cps_occs):.1f}%)")
    print(f"  Employment coverage: {100*emp_coverage:.1f}%")

    # Save
    skill_matrix.to_csv(os.path.join(PROC_DIR, "skill_matrix_by_occ2010.csv"))

    titles_df = pd.DataFrame({
        "occ2010": list(occ2010_titles.keys()),
        "title": list(occ2010_titles.values())
    })
    titles_df.to_csv(os.path.join(PROC_DIR, "occ2010_titles.csv"), index=False)

    # Save crosswalk
    xw_rows = []
    for occ, soc_set in mapping.items():
        for soc in soc_set:
            xw_rows.append({"occ2010": occ, "soc6_onet": soc})
    xw_df = pd.DataFrame(xw_rows)
    xw_df.to_csv(os.path.join(PROC_DIR, "crosswalk_occ2010_to_onet_soc.csv"), index=False)

    print(f"\nDone! Saved to {PROC_DIR}")


if __name__ == "__main__":
    main()

# Progress Report

**Jacob Guzman — Capstone**

---

## 1. Goal from Previous Class Session

My goal was to begin data collection for my capstone project, which examines whether occupations with high AI task exposure and low pre-existing skill portability experienced larger employment declines after the introduction of generative AI tools in late 2022. Specifically, I aimed to identify and acquire the key datasets I need — occupational skill data from O*NET, job-switching data from the Current Population Survey (CPS), and an AI exposure index — and begin processing them into a usable format.

---

## 2. Concrete Artifact: O*NET Skill Data Summary

I downloaded and processed the O*NET 30.1 database (December 2025 release), which provides detailed information on the tasks, skills, and abilities associated with each occupation. I combined three O*NET data files — Skills (35 dimensions), Abilities (52 dimensions), and Knowledge (33 dimensions) — into a single occupation-by-skill-dimension matrix. After aggregating detailed occupation codes to the 6-digit SOC level and normalizing values to a [0, 1] scale, the resulting matrix covers **774 occupations across 120 skill dimensions**.

Below is a code snippet showing the core processing step, along with a summary of the output:

```python
def load_onet_file(filename, category_label):
    """Load an O*NET data file and filter to Level (LV) scale."""
    df = pd.read_csv(os.path.join(RAW_DIR, filename), sep="\t")
    df = df[df["Scale ID"] == "LV"].copy()          # Keep Level scale only
    df = df[df["Recommend Suppress"] != "Y"].copy()  # Drop suppressed
    df["dimension"] = category_label + ": " + df["Element Name"]
    return df[["O*NET-SOC Code", "dimension", "Data Value"]]
```

| Category   | Dimensions | Rows Loaded |
|------------|-----------|-------------|
| Skills     | 35        | 31,143      |
| Abilities  | 52        | 46,420      |
| Knowledge  | 33        | 24,642      |
| **Total**  | **120**   | **102,205** |

**Final matrix:** 774 occupations × 120 skill dimensions (4.9% missing values, filled with 0).

As a sanity check, the top skill dimensions for Chief Executives (SOC 11-1011) are Administration and Management (1.00), Time Management (1.00), and Systems Analysis (0.99) — which is sensible and gives me confidence in the data.

---

## 3. Reflection

The O*NET data collection and processing went smoothly — the database is well-structured and the documentation is clear, which made it straightforward to extract and combine the skill, ability, and knowledge dimensions into a single matrix. One decision I had to make was which O*NET scale to use; I chose the "Level" (LV) scale rather than "Importance" (IM) because level values better capture the degree to which a skill is actually required in practice, which aligns more closely with Khachiyan's (2021) framework for measuring skill portability. The main challenge so far is that I still need to acquire CPS microdata for the job-switching matrix and employment outcomes — I plan to set up an IPUMS CPS extract this week with variables for current and prior-year occupation, employment status, and wages. I also need to obtain the AI exposure index from a classmate. My next step is to get the CPS extract submitted and, while waiting for it, begin writing the code for the skill portability model (the two-step OLS residualization + Random Forest procedure from Khachiyan 2021) so it is ready to run once the switching data arrives.

---

## 4. Question for Peer Feedback

In constructing the skill portability measure, I need to decide how many O*NET dimensions to include as features in the model. I currently use all 120 (Skills + Abilities + Knowledge), but I could also add Work Activities or Work Context data, which would push the feature count above 200. **Do you think using a larger feature set risks overfitting in the Random Forest step, or is it better to include more information and let the model sort out what matters?**

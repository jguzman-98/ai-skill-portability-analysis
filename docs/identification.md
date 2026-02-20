# Research Design: Identification Questions

---

## 1. Treatment and intervention

The treatment is **exposure to generative AI capabilities**, triggered by the public release of ChatGPT in November 2022 and the rapid proliferation of large language model tools that followed. This is not a binary treatment — it varies continuously across occupations based on the degree to which each occupation's tasks can be performed or assisted by AI. I measure treatment intensity using an AI task exposure index (derived from OpenAI data) that captures the share of an occupation's tasks that overlap with current AI capabilities.

Critically, the treatment is not randomly assigned. Occupations were "pre-assigned" different levels of AI exposure based on their existing task composition, which was determined long before generative AI arrived. This is what makes the design feasible: the exposure is a function of pre-existing occupational characteristics, not a response to AI itself.

---

## 2. Main outcome variable

The main outcome is the **percent change in occupation-level employment** between the pre-period (average of 2018, 2019, and 2021) and the post-period (average of 2023–2025), constructed from CPS monthly data. This captures net employment changes at the occupation level — reflecting both job losses from displacement and any offsetting job creation or worker inflows.

I exclude 2020 from the pre-period baseline to avoid conflating COVID-related disruptions with the AI treatment. Alternative outcomes for robustness include log employment changes and, if data permit, occupation-level wage changes.

---

## 3. Comparison/control group?

There is no clean binary control group. Instead, identification comes from **continuous variation in treatment intensity** across occupations. Occupations with low AI task exposure serve as the implicit comparison for those with high exposure.

More precisely, the core test relies on a **double comparison**:

- **First dimension (AI exposure):** High-exposure vs. low-exposure occupations
- **Second dimension (skill portability):** Within high-exposure occupations, those with high vs. low pre-existing skill portability

The main hypothesis is about this second dimension: among AI-exposed occupations, those where workers have more outside options (higher portability) should show smaller employment declines than those where workers are "trapped" (lower portability). Low-exposure occupations, regardless of portability, serve as the baseline against which both groups are compared.

This is conceptually similar to a difference-in-differences framework with continuous treatment intensity and a moderating variable.

---

## 4. Time periods of data

| Period | Years | Role |
|--------|-------|------|
| Extended pre-period | 2010–2017 | Used to construct the switching matrix for the skill portability measure; available for placebo tests |
| Pre-period baseline | 2018–2019, 2021 | Baseline employment levels (2020 excluded due to COVID) |
| Treatment onset | Late 2022 | ChatGPT release (November 2022) |
| Post-period | 2023–2025 | Post-treatment employment outcomes |

The switching matrix for skill portability is constructed by pooling CPS ASEC data across 2010–2025, so the portability measure reflects long-run patterns of occupational mobility rather than short-run post-AI dynamics. This is intentional — portability is meant to capture the *pre-existing structural capacity* for workers to transition, not their response to AI.

The post-period is relatively short (roughly 2.5 years of post-treatment data), which limits the ability to detect longer-run adjustment dynamics and is an acknowledged limitation.

---

## 5. Main threats to identification?

**a. Confounding demand shocks.** Employment changes between the pre- and post-period may reflect demand shifts unrelated to AI — for example, post-COVID recovery patterns, interest rate changes affecting housing/construction, or shifts in consumer spending. If these shocks are correlated with AI exposure, the estimated effect of AI is biased. *Mitigation:* Controlling for pre-trends and broad occupation category fixed effects; the interaction term (AI × portability) is less vulnerable than the main effect because it would require the confound to differentially affect high- vs. low-portability occupations within exposure levels.

**b. Measurement error in AI exposure.** The AI exposure index is an imperfect proxy for actual AI adoption and impact. If it systematically mismeasures exposure for certain occupation types, this introduces attenuation bias or spurious correlations. *Mitigation:* Robustness checks with alternative exposure measures.

**c. Substitution vs. complementarity ambiguity.** AI exposure could lead to job displacement (substitution) or productivity gains that increase labor demand (complementarity). These opposing forces may cancel out in employment data, making it difficult to detect net effects. *Mitigation:* This is partly addressed by the research design — the portability moderator is most relevant for the substitution channel, so the interaction term specifically isolates heterogeneity in the displacement pathway.

**d. Reverse causality in portability.** If workers in AI-exposed occupations began switching occupations *because of* anticipated AI displacement, observed switching patterns would be endogenous to the treatment. *Mitigation:* The portability measure is constructed from switching data pooled across 2010–2025. Since the bulk of this data predates the AI shock, and the portability measure captures structural skill overlap rather than contemporaneous flows, this concern is limited. A stronger approach would be to construct portability using only pre-2022 switching data as a robustness check.

**e. Composition effects.** Changes in occupation-level employment could reflect workers reclassifying across occupation codes rather than actual job losses. *Mitigation:* Using broad occupation categories as controls and verifying that results are not driven by small, volatile occupations.

**f. Limited post-period.** With only ~2.5 years of post-treatment data, short-run effects may differ substantially from long-run equilibrium outcomes. The results should be interpreted as early-stage effects.

---

## 6. Methodology

The setting most naturally fits a **cross-sectional regression with continuous treatment intensity and a moderating variable** — conceptually related to a difference-in-differences design but without a strict binary treatment or parallel trends assumption.

**Why not a standard DiD?** There is no clean treatment/control group — all occupations are exposed to AI to some degree. The "treatment" is continuous (AI exposure intensity), not binary.

**Why not a shift-share / Bartik instrument?** A Bartik approach would require an exogenous component of AI exposure growth, which is difficult to construct here since AI capabilities are endogenous to the technology itself.

**What I use instead:**

The core specification is:

$$\Delta Emp_o = \beta_1 \cdot AIExposure_o + \beta_2 \cdot SkillPort_o + \beta_3 \cdot (AIExposure_o \times SkillPort_o) + X_o'\gamma + \epsilon_o$$

This is a **moderated cross-sectional design** where:
- The *first difference* comes from comparing pre- vs. post-2022 employment within each occupation (so each occupation serves as its own control over time)
- Variation across occupations in AI exposure provides the treatment intensity
- The interaction with portability tests whether the AI-employment relationship is moderated by workers' structural capacity to transition

This approach is appropriate because:
1. **Treatment timing is sharp and common** — generative AI arrived for all occupations simultaneously in late 2022, so there is no staggered adoption concern
2. **Treatment intensity is pre-determined** — AI exposure is based on pre-existing task content, not post-2022 choices
3. **The moderator (portability) is also pre-determined** — constructed from structural skill data and long-run switching patterns, not endogenous to the AI shock
4. **The interaction term is the key test** — even if the main effect of AI exposure is confounded, the differential effect by portability is harder to explain with alternative stories (a confounder would need to differentially affect high- vs. low-portability occupations within exposure groups)

**Limitations of this approach:** It does not establish a causal effect of AI on employment. The design is better suited to answering the *conditional* question — "among occupations affected by AI, does portability predict who adjusts better?" — than the *unconditional* question of whether AI reduces employment overall. This is acknowledged in the paper and is consistent with the stated research goal.


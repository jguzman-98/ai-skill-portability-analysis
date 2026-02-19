# Methodology

## 3.1 Data

This study draws on three primary data sources.

**Occupational skill data** come from the O\*NET 30.1 database (December 2025 release), which provides detailed and regularly updated information on the tasks, skills, and abilities associated with each occupation. I combine three O\*NET data files — Skills (35 dimensions), Abilities (52 dimensions), and Knowledge (33 dimensions) — into a single occupation-by-skill-dimension matrix. Each dimension is measured on the O\*NET "Level" scale, which captures the degree to which each skill is required in practice. After aggregating detailed O\*NET-SOC codes to the 6-digit SOC level and normalizing values to a [0, 1] scale, the resulting matrix covers 774 occupations across 120 skill dimensions.

**Job-switching data** are drawn from the Current Population Survey (CPS), accessed through the Integrated Public Use Microdata Series (IPUMS). I use the CPS Annual Social and Economic Supplement (ASEC) from 2010 to 2025, which records both workers' current occupation and their occupation one year prior. A switch is defined as a difference between these two occupations for workers employed in both periods. Using ASEC person weights, I compute the total weighted flow for each directional occupation pair (e.g., from bus driver to truck driver), yielding a switching matrix of 28,861 directional pairs across 717 origin occupations. From the same data, I compute annual occupation-level employment counts using the full monthly CPS sample to track employment dynamics over time.

**AI task exposure** is measured using [the OpenAI-based AI exposure index, to be described when index is obtained from classmate]. This index captures the degree to which each occupation's tasks can be performed or substantially assisted by current AI and machine learning technologies.

A crosswalk between CPS occupation codes (Census 2010 scheme) and O\*NET SOC codes is constructed using three sources in sequence: the Census Bureau's 2010-to-2018 occupation crosswalk, the IPUMS ACS OCC-to-OCCSOC crosswalk, and title-based fuzzy matching for remaining gaps. Where one CPS code maps to multiple O\*NET occupations, skill vectors are averaged. This procedure achieves 98.7% coverage of total CPS employment.

## 3.2 Measuring Skill Portability

I follow Khachiyan (2021) in constructing an empirically grounded measure of skill portability between occupations. This approach improves on existing measures (Euclidean distance, angular separation, factor analysis) by disciplining skill distance with observed worker switching behavior.

The method proceeds in two steps, building on the Cortes and Gallipoli (2018) framework for occupation switching derived from an Eaton-Kortum-Roy model. In this model, the share of workers switching from occupation $o$ to occupation $o'$, relative to those staying in $o$, can be decomposed into occupation-specific factors and a pair-specific switching cost:

$$\ln\left(\frac{s_{o,o'}}{s_{o,o}}\right) = \alpha_o + \lambda_{o'} + \beta \cdot \text{SkillDist}_{o,o'} + \epsilon_{o,o'}$$

**Step 1: OLS Residualization.** I regress the log switching share on origin and destination occupation fixed effects using iterative demeaning:

$$\ln\left(\frac{s_{o,o'}}{s_{o,o}}\right) = \alpha_o + \lambda_{o'} + \epsilon_{o,o'}$$

The occupation fixed effects explain approximately 51.5% of the variation in switching shares ($R^2 = 0.515$), absorbing occupation-level factors such as size, wage levels, and general desirability. The residuals $\hat{\epsilon}_{o,o'}$ capture the pair-specific component of switching behavior unexplained by individual occupation characteristics.

**Step 2: Random Forest Prediction.** I use a Random Forest algorithm to predict these residuals from the skill characteristics of both occupations in each directional pair. The feature set includes the origin occupation's skill vector, the destination occupation's skill vector, and their element-wise differences — totaling 360 features (120 dimensions $\times$ 3). The Random Forest is trained with 200 trees, maximum depth of 20, and a minimum of 10 samples per leaf.

$$\hat{\epsilon}_{o,o'} = RF\{q_{j,o},\ q_{j,o'},\ q_{j,o} - q_{j,o'}\}$$

The predicted value from this model constitutes the **skill portability** measure: the component of pair-specific switching behavior that is explained by the skill data. In 5-fold cross-validation, the model achieves a mean $R^2$ of 0.08, indicating that skill data explains approximately 8% of the pair-specific residual variation in switching. While modest, this is consistent with Khachiyan (2021), who notes that other pair-specific factors (e.g., geographic overlap, licensing requirements) also contribute to the residual.

The top skill dimensions driving portability, as measured by Random Forest feature importance, are Operations Monitoring, Written Expression, Troubleshooting, and Mechanical Knowledge — suggesting that both cognitive-communicative and technical-manual skill differences are important determinants of worker mobility.

**Aggregation.** To obtain an occupation-level portability score, I aggregate pairwise portability measures across all potential destination occupations, weighted by employment in each destination:

$$P_{o} = \sum_{o' \neq o} P_{o,o'} \cdot \text{employment}_{o'}$$

This captures the overall "outside options" available to workers in occupation $o$ based on skill compatibility with the rest of the labor market. The resulting measure ranges from 0.23 (Dentists — highly specialized, few transferable skills) to 0.58 (Occupational Therapy Assistants — broad skill overlap with many healthcare and service occupations).

## 3.3 Empirical Strategy

The main analysis examines whether pre-existing skill portability moderates the labor market impact of AI exposure following the introduction of generative AI tools in late 2022. The core specification is a cross-sectional regression at the occupation level:

$$\Delta \text{Emp}_o = \beta_1 \cdot \text{AIExposure}_o + \beta_2 \cdot \text{SkillPort}_o + \beta_3 \cdot (\text{AIExposure}_o \times \text{SkillPort}_o) + X_o'\gamma + \epsilon_o$$

where $\Delta \text{Emp}_o$ is the percent change in employment from the pre-period (average of 2018, 2019, and 2021, excluding the COVID-affected year 2020) to the post-period (average of 2023-2025); $\text{AIExposure}_o$ is the occupation-level AI task exposure index; $\text{SkillPort}_o$ is the aggregate skill portability measure; and $X_o$ includes controls for pre-period log employment (occupation size), pre-period employment trend, and broad occupation category fixed effects.

The coefficient of interest is $\beta_3$, the interaction between AI exposure and skill portability. The hypothesis is that $\beta_3 > 0$: among occupations with high AI exposure, those with greater skill portability (more outside options for displaced workers) should experience smaller employment declines, as workers are better able to transition to alternative occupations. A negative $\beta_1$ would indicate that AI-exposed occupations experienced employment declines on average, while $\beta_3 > 0$ would indicate that portability buffers this effect.

This specification does not claim to identify a causal effect of AI on employment. Rather, it asks whether the *heterogeneity* in post-2022 employment outcomes across AI-exposed occupations is systematically related to pre-existing skill portability — a measure constructed entirely from pre-period data and thus not subject to reverse causality from the AI shock itself.

**Robustness checks** include: (i) alternative AI exposure measures; (ii) including the COVID year 2020 in the pre-period; (iii) weighting regressions by pre-period employment; (iv) excluding small occupations; and (v) a placebo test using the pre-2022 period (comparing 2015-2018 to 2018-2021) to verify that the interaction effect is specific to the post-AI period.

# Police_Response

Replication pipeline for *Selective Non-Securitisation: A Cross-National Quantitative Analysis of Police Response to Political Protest in France, Germany, Italy, Spain and the United Kingdom* (K. Nutley, University of St Andrews).

This README lays out the pipeline in run order. 

## Pipeline order

```
1. data_collection/01_ACLED_api.R
   → pulls raw ACLED protest events (2020–2025) for FR/DE/IT/ES/UK via authenticated,
     paginated API calls; was done iteratively
   → output: acled_all_countries_combined.csv
2. partisan_classification/{country}/
   ├── {country}_acled_partisan_classification.R # script classifying actor partisanship
   └── {country}_acled_partisan_classification_bootstrapped.R # script applying BART MNLI
3a. data/{country}/
   ├── {country}_acled_partisan_classification.csv # original event data w/ classified partisans
   ├── {country}_acled_classified_actors.csv # discrete actors and their classifications
   ├── {country}_acled_partisan_classification_bootstrapped_pre.csv ##### YOU NEED TO ADD THESE 
   └── {country}_acled_partisan_classification_bootstrapped_post.csv
3b. data/combined/
   ├── combine_classed_countries.R # script combining classified events
   └── acled_all_countries_combined.csv # entire event sample

3. 

2. data_collection/spatial_controls/
   ├── 02a_government_building_distance.R   (osmextract + BART-MNLI feature filtering)
   ├── 02b_major_road_distance.R            (osmextract, motorway/trunk/primary/secondary)
   └── 02c_urban_rural_nuts3.R              (giscoR, NUTS-3 URBN_TYPE join)
   → output: data/intermediate/{country}_spatial_controls.csv

3. partisan_classification/{country}/
   ├── 01_{country}_partisan_classification.R      (Steps 1–4: CPDS gov / PPDB party /
   │                                                  manual ideological / union rules)
   ├── 02_{country}_zero_shot_bootstrapped.py       (Step 5: BART-MNLI zero-shot NLI,
   │                                                  iterative correction — versioned,
   │                                                  see iteration table below)
   └── outputs:
       {country}_acled_classified_actors.csv
       {country}_acled_partisan_classification.csv
       {country}_acled_partisan_classification_bootstrappedN.csv   (final = v7 UK / v8 DE /
                                                                     v5 FR / v6 ES / v3 IT)

4. data_collection/event_descriptive_stats.R
   → partisan breakdown table (Table: partisan_breakdown), goal-category distribution

5. response_classification/
   ├── 01_classifying_binary_response.R    (7-step exclusion/detection rule engine)
   ├── 02_roberta_severity.py              (RoBERTa-base multi-label fine-tune:
   │                                         presence / arrest / brutality)
   └── outputs:
       acled_all_countries_combined_classed1.csv
       acled_classified_police_presence1.csv
       acled_police_response_subset1.csv

6. analysis/   [NEW — not yet in repo, added to match Section 5/6 of the manuscript]
   ├── 01_model1_police_presence.R      (logistic regression, H1/H3, country FE)
   ├── 02_heckman_selection.R           (first-stage probit + IMR, exclusion restriction)
   ├── 03_model2_severity_heckman.R     (arrest / brutality outcome equations)
   ├── 04_tost_equivalence.R            (H2 equivalence testing, δ = 0.184)
   └── 05_robustness_clustered_se.R     (admin2-clustered SE robustness check)
```

## Iteration versions (zero-shot classification)

| Country | Prompt-only version | Corrected version | Accuracy (prompt-only → corrected) |
|---|---|---|---|
| France | v3 | v5 | 92% → 97% |
| Germany | v4 | v8 | 89% → 93% |
| Italy | v3 | v3 (no correction round) | 95% |
| Spain | v3 | v6 | 87% → 94% |
| UK | v3 | v7 | 83% → 92% |

## Data access note

ACLED introduced a one-year access lag for non-institutional/junior researchers. Initial
pull: October 2025 (Jan 2020–Oct 2024). Supplementary pulls through 2026 to complete
2025 coverage. **2025 figures are provisional** until the full-year lag clears
(expected ~Jan 2027) — re-run `01_ACLED_api.R` and downstream steps before final
submission if the numbers need to be locked.

## Requirements

- R (≥4.2): `tidyverse`, `osmextract`, `sf`, `giscoR`, `sampleSelection` (Heckman), `TOSTER`
- Python (≥3.10): `transformers`, `torch`, `pandas`, `scikit-learn`
- ACLED API credentials (`.Renviron` / `.env`, not committed — see `.gitignore`)

## Status

This pipeline scaffold reflects the manuscript as of the current draft. Sections 6–7
(Discussion, theory reconnection) are still outline-stage in the paper itself and don't
require additional scripts — they're interpretive, not computational.

# Police_Response

Replication pipeline for *Selective Non-Securitisation: A Cross-National Quantitative Analysis of Police Response to Political Protest in France, Germany, Italy, Spain and the United Kingdom* (K. Nutley, University of St Andrews).

This README lays out the pipeline in run order. It's assembled from the script names and
step descriptions scattered through the manuscript's working comments, mapped onto the
repo's existing folder structure (`data_collection/`, `partisan_classification/`,
`response_classification/`, `data/`). Two folders — `analysis/` and
`data_collection/spatial_controls/` — are new; they cover steps the paper describes
(the Heckman/TOST models, the OSM/NUTS-3 controls) that weren't yet named as scripts
in the repo. Flagged inline below wherever I've had to infer content rather than
reproduce something explicitly named.

**Every file in this pipeline is a scaffold, not a working replication.** I don't have
your actual code — I only have what your comments and methods section describe. Each
script below has the correct inputs/outputs, function signatures, and parameters as
specified in the manuscript, with `# TODO` markers wherever the actual logic needs to
be pasted in from your working files. Treat this as the skeleton to hang your existing
scripts on, not a from-scratch reimplementation.

## Pipeline order

```
1. data_collection/01_ACLED_api.R
   → pulls raw ACLED protest events (2020–2025) for FR/DE/IT/ES/UK via authenticated,
     paginated API calls
   → output: data/raw/acled_raw_{country}.csv

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

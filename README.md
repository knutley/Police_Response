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
   ├── {country}_acled_partisan_classification_bootstrapped_pre.csv # pre-NLP classifications 
   └── {country}_acled_partisan_classification_bootstrapped_post.csv # post-NLP classifications
3b. data/combined/
   ├── acled_all_countries_combined_classed1.csv # Partisan classification complete; still to do response
   ├── acled_classified_police_presence1.csv # Rule-based classification of police presence (I build on this)
   ├── acled_classified_severity1.csv # This is the output of the RoBERTa classifier
   └── acled_merged_controls.csv # Final built dataset 
4. response_classification/
    ├── classifying_binary_response.R # algorithmic detection of additional police responses
    ├── response_pos_set.xlsx # positive set 
    ├── neg_pos_set.xlsx # negative set 
    └── roberta_severity.py # builds on the algorithmic detection to find presence, arrest, and brutality
5. control_variable/
    └── control_variable_construction.R
6. analysis/
    ├── model1_presence.R       (logistic regression, H1/H3, country FE)
    ├── model2_severity.R       (arrest / brutality outcome equations)
    ├── heckman_correction.R    (first-stage probit + IMR, exclusion restriction)
    ├── 04_tost_equivalence.R            (H2 equivalence testing, δ = 0.184)
    └── 05_robustness_clustered_se.R     (admin2-clustered SE robustness check)
```

# Heckman Selection Models: Police Response Severity
# Author: Katie Nutley
# Date: 08-06-2026
# Corrected version: fixes sel_formula ordering bug, updates delta
# justification comment, adds no-correction robustness check.

# Two-stage Heckman correction for sample selection bias:
# Selection equation: police presence (observed for all events)
# Outcome equations:  arrest and brutality (observed only when police present)
# Exclusion restriction: n_police_stations_5km — number of police stations
# within 5km radius. Affects probability of police deployment (capacity) but
# has no direct effect on severity of response once police are present.
# Instrument strength: chi-sq = 116, p < 0.001.

# ============================================================================
# PACKAGES
# ============================================================================

library(tidyverse)
library(dplyr)
library(sampleSelection)
library(boot)
library(car)

# ============================================================================
# DATA PREPARATION
# ============================================================================

acled_data <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

cat("n_police_stations_5km present:",
    "n_police_stations_5km" %in% names(acled_data), "\n")
cat("Missing values:", sum(is.na(acled_data$n_police_stations_5km)), "\n")

# ============================================================================
# PARTISAN TYPE DUMMIES, COVID INDICATOR, REFERENCE CATEGORIES
# ============================================================================

acled_data <- acled_data %>%
  mutate(
    counter_protest = as.integer(
      event_partisan_type_final %in% c("left_right", "left_centre", "right_centre")
    ),
    left_pure    = as.integer(event_partisan_type_final == "left"),
    right_pure   = as.integer(event_partisan_type_final == "right"),
    unknown_pure = as.integer(event_partisan_type_final == "unknown"),
    # centre omitted as reference category
    covid        = as.integer(year %in% c(2020, 2021)),
    crowd_size_cat = relevel(factor(crowd_size_cat), ref = "No report"),
    urb_rur        = relevel(factor(urb_rur),        ref = "Predominantly urban"),
    log_dist_govt_building = log1p(dist_govt_building_m)
  )

# Sanity checks
table(acled_data$counter_protest)
table(acled_data$police_presence_bin)
cat("n_police_stations_5km missing:",
    sum(is.na(acled_data$n_police_stations_5km)), "\n")
cat("Arrest N (among police-present):",
    sum(acled_data$arrest[acled_data$police_presence_bin == 1], na.rm = TRUE), "\n")
cat("Brutality N (among police-present):",
    sum(acled_data$brutality[acled_data$police_presence_bin == 1], na.rm = TRUE), "\n")

# ============================================================================
# DESCRIPTIVE: SEVERITY RATES AMONG POLICE-PRESENT EVENTS
# ============================================================================

acled_data %>%
  filter(police_presence_bin == 1) %>%
  group_by(event_partisan_type_final) %>%
  summarise(
    n_events      = n(),
    pct_arrest    = mean(arrest,    na.rm = TRUE) * 100,
    pct_brutality = mean(brutality, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  arrange(desc(pct_arrest))

# ============================================================================
# FORMULAS
# ============================================================================
# NOTE: defined here, up front, so that all downstream diagnostics
# (missingness check, instrument test, Heckman models, bootstrap, and the
# no-correction comparison) reference the same, single source of truth.
# Previously sel_formula was only defined inside the bootstrap section,
# which caused the missingness-diagnostic block below to fail with
# "object 'sel_formula' not found" when run top-to-bottom.

sel_formula <- police_presence_bin ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  urb_rur + covid + protestor_violence +
  n_police_stations_5km

arr_formula <- arrest ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  urb_rur + covid + protestor_violence

brut_formula <- brutality ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  urb_rur + covid + protestor_violence

# ============================================================================
# MISSINGNESS DIAGNOSTIC (now runs correctly — sel_formula exists above)
# ============================================================================

colSums(is.na(acled_data[, all.vars(sel_formula)]))

n_dropped <- nrow(acled_data) - nrow(na.omit(acled_data[, all.vars(sel_formula)]))
cat("\nObservations dropped due to listwise deletion:", n_dropped, "\n")

acled_data$missing_spatial <- is.na(acled_data$log_dist_govt_building) |
  is.na(acled_data$urb_rur) |
  is.na(acled_data$crowd_size_cat)

cat("\n=== MISSINGNESS BY POLICE PRESENCE ===\n")
print(table(acled_data$missing_spatial, acled_data$police_presence_bin))
print(prop.table(table(acled_data$missing_spatial, acled_data$police_presence_bin), margin = 1))

# Report in text: N observations excluded due to missing spatial covariates;
# excluded cases show a lower rate of police presence than retained cases,
# consistent with geocoding failures concentrated in areas with weaker
# administrative infrastructure visibility. Confirm these numbers against
# the table/prop.table output above before citing in the paper — do not
# reuse numbers from an earlier session without re-verifying against this
# run.

# ============================================================================
# INSTRUMENT STRENGTH: VERIFY BEFORE PROCEEDING
# ============================================================================

first_stage <- glm(
  sel_formula,
  data   = acled_data,
  family = binomial(link = "probit")
)

cat("\n=== INSTRUMENT STRENGTH TEST ===\n")
linearHypothesis(first_stage, "n_police_stations_5km = 0")

cat("\nCoefficient on n_police_stations_5km:\n")
print(summary(first_stage)$coefficients["n_police_stations_5km", ])

# ============================================================================
# HECKMAN 1: ARREST
# ============================================================================

cat("\n=== HECKMAN MODEL 1: ARREST ===\n")

heck_arrest <- heckit(
  selection = sel_formula,
  outcome   = arr_formula,
  data      = acled_data,
  method    = "2step"
)

summary(heck_arrest)

# ============================================================================
# HECKMAN 2: BRUTALITY
# ============================================================================

cat("\n=== HECKMAN MODEL 2: BRUTALITY ===\n")

heck_brutality <- heckit(
  selection = sel_formula,
  outcome   = brut_formula,
  data      = acled_data,
  method    = "2step"
)

summary(heck_brutality)

# ============================================================================
# IMR DIAGNOSTICS
# ============================================================================
# Inverse Mills Ratio significance tests whether Heckman correction is
# needed. Non-significant IMR = selection bias not a material concern for
# that outcome.

cat("\n=== IMR DIAGNOSTICS ===\n")
cat("Arrest   — rho:", round(heck_arrest$rho,    4),
    " | IMR p-value:",
    round(summary(heck_arrest)$estimate["invMillsRatio", "Pr(>|t|)"], 4), "\n")

cat("Brutality — rho:", round(heck_brutality$rho, 4),
    " | IMR p-value:",
    round(summary(heck_brutality)$estimate["invMillsRatio", "Pr(>|t|)"], 4), "\n")

# ============================================================================
# BOOTSTRAPPED STANDARD ERRORS
# ============================================================================
# sandwich::vcovHC does not support sampleSelection objects.
# Bootstrap SEs are the standard alternative.
# Use R = 200 for quick checks, R = 500 for final results.

boot_heck <- function(data, indices, selection_formula, outcome_formula) {
  d <- data[indices, ]
  fit <- tryCatch(
    heckit(
      selection = selection_formula,
      outcome   = outcome_formula,
      data      = d,
      method    = "2step"
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) return(rep(NA_real_, length(coef(heck_arrest))))
  coef(fit)
}

# --- Arrest bootstrap ---
cat("\nRunning bootstrap for arrest model (R = 500)...\n")
set.seed(42)
boot_arrest <- boot(
  data      = acled_data,
  statistic = boot_heck,
  R         = 500,
  selection_formula = sel_formula,
  outcome_formula   = arr_formula
)

boot_se_arrest        <- apply(boot_arrest$t, 2, sd, na.rm = TRUE)
names(boot_se_arrest) <- names(coef(heck_arrest))
coefs_arrest          <- coef(heck_arrest)
z_arrest              <- coefs_arrest / boot_se_arrest
p_arrest              <- 2 * pnorm(-abs(z_arrest))

cat("\n=== ARREST: Full results with bootstrap SEs ===\n")
arrest_results <- data.frame(
  Estimate  = round(coefs_arrest,   4),
  Std_Error = round(boot_se_arrest, 4),
  z_value   = round(z_arrest,       4),
  p_value   = round(p_arrest,       4)
)
print(arrest_results)

# --- Brutality bootstrap ---
cat("\nRunning bootstrap for brutality model (R = 500)...\n")
set.seed(42)
boot_brutality <- boot(
  data      = acled_data,
  statistic = boot_heck,
  R         = 500,
  selection_formula = sel_formula,
  outcome_formula   = brut_formula
)

boot_se_brutality        <- apply(boot_brutality$t, 2, sd, na.rm = TRUE)
names(boot_se_brutality) <- names(coef(heck_brutality))
coefs_brutality           <- coef(heck_brutality)
z_brutality               <- coefs_brutality / boot_se_brutality
p_brutality               <- 2 * pnorm(-abs(z_brutality))

cat("\n=== BRUTALITY: Full results with bootstrap SEs ===\n")
brutality_results <- data.frame(
  Estimate  = round(coefs_brutality,   4),
  Std_Error = round(boot_se_brutality, 4),
  z_value   = round(z_brutality,       4),
  p_value   = round(p_brutality,       4)
)
print(brutality_results)

# ============================================================================
# PARTISAN COEFFICIENTS: OUTCOME EQUATIONS SUMMARY
# ============================================================================

names(coefs_arrest)

# outcome_idx locates the outcome-equation block of partisan dummies
# (left_pure, right_pure, unknown_pure, counter_protest). This relies on
# left_pure appearing exactly twice in coefs_arrest — once in the
# selection equation, once in the outcome equation — in that fixed order.
# If the formulas above are ever reordered, re-check this index before
# trusting downstream output.
outcome_idx <- which(names(coefs_arrest) == "left_pure")[2] + 0:3

heck_comparison <- bind_rows(
  data.frame(
    Variable = c("Left", "Right", "Unknown", "Counter-Protest"),
    Outcome  = "Arrest",
    Estimate = round(coefs_arrest[outcome_idx],   4),
    SE       = round(boot_se_arrest[outcome_idx], 4),
    p_value  = round(p_arrest[outcome_idx],       4)
  ),
  data.frame(
    Variable = c("Left", "Right", "Unknown", "Counter-Protest"),
    Outcome  = "Brutality",
    Estimate = round(coefs_brutality[outcome_idx],   4),
    SE       = round(boot_se_brutality[outcome_idx], 4),
    p_value  = round(p_brutality[outcome_idx],       4)
  )
)

cat("\n=== PARTISAN COEFFICIENTS: OUTCOME EQUATIONS ===\n")
print(heck_comparison, row.names = FALSE)

# ============================================================================
# ROBUSTNESS: PLAIN LOGIT WITHOUT HECKMAN CORRECTION
# ============================================================================
# Purpose: the exclusion restriction (police station density) is defended
# on conceptual rather than statistical grounds, since exclusion cannot be
# tested directly. As a check on whether the paper's conclusions depend on
# accepting that restriction, the outcome equations are re-estimated on the
# police-present subsample WITHOUT the inverse Mills ratio / Heckman
# correction (i.e., a plain logistic regression). If the partisan
# coefficients are substantively similar with and without the correction,
# this suggests the null result for H2 is not an artefact of the selection
# correction itself.

cat("\n=== ROBUSTNESS: NO-CORRECTION COMPARISON ===\n")

logit_arrest_nocorr <- glm(
  arr_formula,
  data   = acled_data %>% filter(police_presence_bin == 1),
  family = binomial(link = "logit")
)

logit_brutality_nocorr <- glm(
  brut_formula,
  data   = acled_data %>% filter(police_presence_bin == 1),
  family = binomial(link = "logit")
)

compare_coefs <- function(heck_coefs, heck_p, logit_model, label) {
  logit_coefs <- summary(logit_model)$coefficients
  data.frame(
    Variable  = c("Left", "Right"),
    Outcome   = label,
    Heck_Est  = round(heck_coefs[c("left_pure", "right_pure")], 4),
    Heck_p    = round(heck_p[c("left_pure", "right_pure")], 4),
    Logit_Est = round(logit_coefs[c("left_pure", "right_pure"), "Estimate"], 4),
    Logit_p   = round(logit_coefs[c("left_pure", "right_pure"), "Pr(>|z|)"], 4)
  )
}

comparison_table <- bind_rows(
  compare_coefs(coefs_arrest[outcome_idx],
                setNames(p_arrest[outcome_idx], c("left_pure", "right_pure", "unknown_pure", "counter_protest")),
                logit_arrest_nocorr, "Arrest"),
  compare_coefs(coefs_brutality[outcome_idx],
                setNames(p_brutality[outcome_idx], c("left_pure", "right_pure", "unknown_pure", "counter_protest")),
                logit_brutality_nocorr, "Brutality")
)

# NOTE: compare_coefs() as written indexes heck_coefs/heck_p by name
# ("left_pure", "right_pure"), but outcome_idx-subsetted vectors above
# already carry those names from coefs_arrest / p_arrest. Confirm names()
# on coefs_arrest[outcome_idx] before running — if names come through as
# "left_pure","right_pure","unknown_pure","counter_protest" this works
# directly; if not, assign names explicitly first, e.g.:
#   names(coefs_arrest)[outcome_idx] <- c("left_pure","right_pure","unknown_pure","counter_protest")

print(comparison_table, row.names = FALSE)
write.csv(comparison_table, "results/heckman_vs_nocorrection.csv", row.names = FALSE)

# ============================================================================
# TOST: EQUIVALENCE TESTING FOR H2 (PARTISAN NULL, OUTCOME EQUATIONS)
# ============================================================================
# TOST tests whether an estimate falls within a pre-specified equivalence
# bound [-delta, +delta], rather than merely failing to reject beta = 0.
# Two one-sided z-tests against the bounds; both must be significant
# (p < .05) for the estimate to be declared "equivalent to zero" at that
# bound, which corresponds to the 90% CI falling entirely within
# [-delta, +delta].

tost_z <- function(estimate, se, delta) {
  z_lower <- (estimate - (-delta)) / se
  z_upper <- (estimate - delta) / se
  p_lower <- pnorm(z_lower, lower.tail = FALSE)  # test against -delta
  p_upper <- pnorm(z_upper, lower.tail = TRUE)   # test against +delta
  p_tost  <- max(p_lower, p_upper)               # TOST p-value = larger of the two
  ci90_lo <- estimate - 1.645 * se
  ci90_hi <- estimate + 1.645 * se
  equivalent <- (p_tost < 0.05)
  data.frame(
    estimate   = round(estimate, 4),
    se         = round(se, 4),
    delta      = delta,
    p_lower    = round(p_lower, 4),
    p_upper    = round(p_upper, 4),
    p_tost     = round(p_tost, 4),
    ci90_lower = round(ci90_lo, 4),
    ci90_upper = round(ci90_hi, 4),
    equivalent = equivalent
  )
}

# ----------------------------------------------------------------------
# Choose delta: design-based / achieved 80%-power bound, derived from each
# coefficient's standard error alone (NOT its point estimate — this
# avoids circularity, since delta is a function of sample precision, not
# of the result being tested). delta_primary is set to the LARGEST of the
# four achieved 80%-power bounds across the partisan coefficients of
# interest (arrest/right = 0.1836), ensuring all four TOST tests below are
# adequately powered at this single, shared threshold.
#
# An earlier version of this analysis anchored delta to half the Model 1
# left-right deployment-stage effect (0.414 log-odds / 2 = 0.207). That
# approach was abandoned: Model 1 is estimated via logistic regression
# (log-odds scale), while heckit()'s outcome equations are estimated via
# OLS (linear probability scale) under method = "2step". Anchoring a
# linear-probability-scale delta to a log-odds-scale effect size mixes
# units and is not defensible without an explicit conversion (e.g., via
# average marginal effects). The SE-derived bound below avoids this
# problem entirely.
#
# Rule of thumb for 80% power TOST: delta_80 ≈ SE * (1.645 + 0.84) ≈ SE * 2.49
# ----------------------------------------------------------------------

delta_primary <- 0.18355

achieved_delta <- function(se) round(se * 2.49, 4)

cat("\n=== TOST: ARREST OUTCOME EQUATION ===\n")
tost_arrest <- bind_rows(
  cbind(Variable = "Left",  tost_z(coefs_arrest[outcome_idx[1]], boot_se_arrest[outcome_idx[1]], delta_primary)),
  cbind(Variable = "Right", tost_z(coefs_arrest[outcome_idx[2]], boot_se_arrest[outcome_idx[2]], delta_primary))
)
tost_arrest$achieved_delta_80pct <- c(
  achieved_delta(boot_se_arrest[outcome_idx[1]]),
  achieved_delta(boot_se_arrest[outcome_idx[2]])
)
print(tost_arrest, row.names = FALSE)

cat("\n=== TOST: BRUTALITY OUTCOME EQUATION ===\n")
tost_brutality <- bind_rows(
  cbind(Variable = "Left",  tost_z(coefs_brutality[outcome_idx[1]], boot_se_brutality[outcome_idx[1]], delta_primary)),
  cbind(Variable = "Right", tost_z(coefs_brutality[outcome_idx[2]], boot_se_brutality[outcome_idx[2]], delta_primary))
)
tost_brutality$achieved_delta_80pct <- c(
  achieved_delta(boot_se_brutality[outcome_idx[1]]),
  achieved_delta(boot_se_brutality[outcome_idx[2]])
)
print(tost_brutality, row.names = FALSE)

tost_results <- bind_rows(
  cbind(Outcome = "Arrest",    tost_arrest),
  cbind(Outcome = "Brutality", tost_brutality)
)
write.csv(tost_results, "results/tost_equivalence.csv", row.names = FALSE)

# ============================================================================
# SAVE RESULTS
# ============================================================================

write.csv(heck_comparison,   "results/heckman_partisan_coefs.csv", row.names = FALSE)
write.csv(arrest_results,    "results/heckman_arrest_full.csv",    row.names = FALSE)
write.csv(brutality_results, "results/heckman_brutality_full.csv", row.names = FALSE)
saveRDS(heck_arrest,    "models/heckman_arrest.rds")
saveRDS(heck_brutality, "models/heckman_brutality.rds")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to results/ and models/ directories\n")
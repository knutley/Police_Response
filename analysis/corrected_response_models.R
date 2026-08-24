# Heckman Selection Models: Police Response Severity
# Author: Katie Nutley
# Date: 08-06-2026
# Corrected version (20-08-2026): fixes sel_formula ordering bug, updates
# delta justification comment, adds no-correction robustness check
# [carried over from prior correction], AND adds log_dist_major_road +
# is_weekend to sel_formula / arr_formula / brut_formula to match the
# corrected Model 1 specification (01_logit_police_presence_CORRECTED.R).
#
# Model 1 and Model 2 share the same right-hand-side control set except for
# the exclusion restriction (n_police_stations_5km, selection equation only)
# and protestor_violence (was already present here, absent from Model 1's
# baseline by design — it enters Model 1 as a separate robustness step).
# Keeping the two specifications aligned matters because Section 6.2 frames
# Model 2 as "parallel" to Model 1 in its predictor set.

# Two-stage Heckman correction for sample selection bias:
# Selection equation: police presence (observed for all events)
# Outcome equations:  arrest and brutality (observed only when police present)
# Exclusion restriction: n_police_stations_5km — number of police stations
# within 5km radius. Affects probability of police deployment (capacity) but
# has no direct effect on severity of response once police are present.
# Instrument strength: chi-sq = 116, p < 0.001 (RE-VERIFY below — this will
# shift slightly now that two more covariates are in the selection equation).

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
cat("Missing values (n_police_stations_5km):",
    sum(is.na(acled_data$n_police_stations_5km)), "\n")
cat("is_weekend present:", "is_weekend" %in% names(acled_data), "\n")
cat("log_dist_major_road present:",
    "dist_major_road_m" %in% names(acled_data) ||
      "log_dist_major_road" %in% names(acled_data), "\n")

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
    log_dist_govt_building = log1p(dist_govt_building_m),
    log_dist_major_road    = log1p(dist_major_road_m)
  )

# Sanity checks
table(acled_data$counter_protest)
table(acled_data$police_presence_bin)
table(acled_data$is_weekend)
cat("n_police_stations_5km missing:",
    sum(is.na(acled_data$n_police_stations_5km)), "\n")
cat("log_dist_major_road missing:",
    sum(is.na(acled_data$log_dist_major_road)), "\n")
cat("is_weekend missing:",
    sum(is.na(acled_data$is_weekend)), "\n")
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
# Use this table to rebuild Table 10's raw N/rate columns directly from the
# category-level rates — do not hand-copy numbers from an earlier run.

# ============================================================================
# FORMULAS
# ============================================================================
# NOTE: defined here, up front, so that all downstream diagnostics
# (missingness check, instrument test, Heckman models, bootstrap, and the
# no-correction comparison) reference the same, single source of truth.
#
# CORRECTED: log_dist_major_road and is_weekend added to all three formulas,
# to match the corrected Model 1 specification and the Section 6.1 claim
# that both variables are used as controls throughout.

sel_formula <- police_presence_bin ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  log_dist_major_road + urb_rur + covid + is_weekend +
  protestor_violence +
  n_police_stations_5km

arr_formula <- arrest ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  log_dist_major_road + urb_rur + covid + is_weekend +
  protestor_violence

brut_formula <- brutality ~
  left_pure + right_pure + unknown_pure + counter_protest +
  country + crowd_size_cat + log_dist_govt_building +
  log_dist_major_road + urb_rur + covid + is_weekend +
  protestor_violence

# ============================================================================
# MISSINGNESS DIAGNOSTIC
# ============================================================================
# CORRECTED: missing_spatial now also checks log_dist_major_road. The
# previously-reported "1,490 observations dropped" figure will change once
# this and the two new formula terms are included — re-verify against this
# run's output before citing a number in the manuscript.

colSums(is.na(acled_data[, all.vars(sel_formula)]))

n_dropped <- nrow(acled_data) - nrow(na.omit(acled_data[, all.vars(sel_formula)]))
cat("\nObservations dropped due to listwise deletion:", n_dropped, "\n")

acled_data$missing_spatial <- is.na(acled_data$log_dist_govt_building) |
  is.na(acled_data$log_dist_major_road) |
  is.na(acled_data$urb_rur) |
  is.na(acled_data$crowd_size_cat)

cat("\n=== MISSINGNESS BY POLICE PRESENCE ===\n")
print(table(acled_data$missing_spatial, acled_data$police_presence_bin))
print(prop.table(table(acled_data$missing_spatial, acled_data$police_presence_bin), margin = 1))

# Report in text: N observations excluded due to missing spatial covariates;
# confirm the dropped-N and the presence-rate comparison against THIS run's
# output before citing in the paper. Do not reuse numbers from any earlier
# session (including the pre-correction run) without re-verifying here.

# ============================================================================
# INSTRUMENT STRENGTH: VERIFY BEFORE PROCEEDING
# ============================================================================
# The chi-sq = 116 figure quoted at the top of this script was estimated
# under the PRE-correction sel_formula (without log_dist_major_road /
# is_weekend). Re-run and re-report from this output.

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
# that outcome. Re-verify the IMR / rho values quoted in Section 7.2 against
# this output — they were estimated under the pre-correction spec.

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
# IMPORTANT: print and manually inspect this before trusting outcome_idx
# below. Adding log_dist_major_road and is_weekend changes the total number
# of terms in each equation, but NOT the ordering logic (left_pure still
# appears once in the selection block and once in the outcome block, in that
# order) — outcome_idx should still resolve correctly, but confirm the four
# names it pulls out actually read "left_pure","right_pure","unknown_pure",
# "counter_protest" before proceeding.

outcome_idx <- which(names(coefs_arrest) == "left_pure")[2] + 0:3
stopifnot(identical(
  names(coefs_arrest)[outcome_idx],
  c("left_pure", "right_pure", "unknown_pure", "counter_protest")
))

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
# This is the table underlying manuscript Tables 9 and 10's beta/p columns.
# Rebuild those tables' N / raw-rate columns from the descriptive block above
# (grouped by event_partisan_type_final), NOT by hand-copying prior numbers —
# Table 10's Counter-Protest row was wrong in the last version for exactly
# this reason.

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
# of the result being tested).
#
# IMPORTANT: delta_primary = 0.18355 below is a HARD-CODED CARRYOVER from
# the pre-correction run (arrest/right SE-derived bound under the OLD
# sel_formula). Adding log_dist_major_road and is_weekend changes every
# coefficient's SE, which changes each achieved_delta() value, which means
# the LARGEST of the four achieved bounds may no longer be 0.18355, and may
# not even still be the arrest/right coefficient. Recompute delta_primary
# from this run's actual bootstrap SEs before using it below — do not reuse
# the old constant.
# ----------------------------------------------------------------------

achieved_delta <- function(se) round(se * 2.49, 4)

achieved_bounds <- c(
  arrest_left    = achieved_delta(boot_se_arrest[outcome_idx[1]]),
  arrest_right   = achieved_delta(boot_se_arrest[outcome_idx[2]]),
  brutality_left = achieved_delta(boot_se_brutality[outcome_idx[1]]),
  brutality_right= achieved_delta(boot_se_brutality[outcome_idx[2]])
)
print(achieved_bounds)

delta_primary <- max(achieved_bounds)
cat("\ndelta_primary (recomputed, largest achieved 80%-power bound):",
    delta_primary,
    "\n  -> driven by:", names(which.max(achieved_bounds)), "\n")
# Report this recomputed value (and which coefficient drives it) in
# Section 6.2.3 in place of the old 0.18355 figure and its "arrest/right"
# attribution — both may have changed.

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
# TABLE 9 / TABLE 10 REBUILD HELPER
# ============================================================================
# Produces the raw N / rate columns for Tables 9 and 10 directly from the
# data, joined to the regression results above, so the descriptive columns
# can never again drift out of sync with heck_comparison (as happened with
# the previous Table 10 counter-protest row).

severity_desc <- acled_data %>%
  filter(police_presence_bin == 1) %>%
  mutate(
    partisan_group = case_when(
      event_partisan_type_final == "centre" ~ "Centre",
      left_pure == 1                        ~ "Left",
      right_pure == 1                       ~ "Right",
      unknown_pure == 1                     ~ "Unknown",
      counter_protest == 1                  ~ "Counter-Protest",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(partisan_group)) %>%
  group_by(partisan_group) %>%
  summarise(
    N            = n(),
    Arrests      = sum(arrest, na.rm = TRUE),
    Arrest_Rate  = round(mean(arrest, na.rm = TRUE) * 100, 2),
    Brutal       = sum(brutality, na.rm = TRUE),
    Brutal_Rate  = round(mean(brutality, na.rm = TRUE) * 100, 2),
    .groups = "drop"
  )

print(severity_desc)
write.csv(severity_desc, "results/table9_table10_descriptive_columns.csv", row.names = FALSE)
# Use this file's Arrests/Arrest_Rate for Table 9's N/Rate columns, and
# Brutal/Brutal_Rate for Table 10's N/Rate columns, for every partisan
# category including Counter-Protest. Merge with heck_comparison (beta, p,
# TOST) by Variable/partisan_group to assemble the final tables.

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
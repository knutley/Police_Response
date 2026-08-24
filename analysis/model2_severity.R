# Heckman Selection Models: Police Response Severity
# Author: Katie Nutley
# Date: 08-06-2026

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

# Reload the data — should now have n_police_stations_5km in it
acled_data <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

# Check it's there
cat("n_police_stations_5km present:", 
    "n_police_stations_5km" %in% names(acled_data), "\n")
cat("Missing values:", sum(is.na(acled_data$n_police_stations_5km)), "\n")

# I didn't save this, because I'm a goose 

# Check if the spatial objects are still there
exists("acled_sf_m")
exists("police_stations")
exists("n_stations")

# Right, n_stations still exists so that's good 

acled_data <- acled_data %>%
  left_join(
    n_stations,
    by = "event_id_cnty"
  ) %>%
  mutate(
    n_police_stations_5km = replace_na(n_police_stations_5km, 0),
    n_police_stations_5km = ifelse(
      country == "Spain" & longitude < -10,
      NA,
      n_police_stations_5km
    )
  )

# Verify
cat("Present:", "n_police_stations_5km" %in% names(acled_data), "\n")
cat("Missing:", sum(is.na(acled_data$n_police_stations_5km)), "\n") # I removed the Canary Islands lol 

# Save
write_csv(acled_data,
          "~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

# Recreate dummies and rerun instrument test
acled_data <- acled_data %>%
  mutate(
    counter_protest = as.integer(
      event_partisan_type_final %in% c("left_right", "left_centre", "right_centre")
    ),
    left_pure    = as.integer(event_partisan_type_final == "left"),
    right_pure   = as.integer(event_partisan_type_final == "right"),
    unknown_pure = as.integer(event_partisan_type_final == "unknown"),
    covid        = as.integer(year %in% c(2020, 2021)),
    crowd_size_cat = relevel(factor(crowd_size_cat), ref = "No report"),
    urb_rur        = relevel(factor(urb_rur),        ref = "Predominantly urban"),
    log_dist_govt_building = log1p(dist_govt_building_m)
  )


# Set reference categories
acled_data <- acled_data %>%
  mutate(
    crowd_size_cat = relevel(factor(crowd_size_cat), ref = "No report"),
    urb_rur        = relevel(factor(urb_rur),        ref = "Predominantly urban"),
    
    # Log-transform distance variables
    log_dist_govt_building = log1p(dist_govt_building_m)
  )

# ============================================================================
# PARTISAN TYPE DUMMIES + COVID INDICATOR
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
    covid        = as.integer(year %in% c(2020, 2021))
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
# INSTRUMENT STRENGTH: VERIFY BEFORE PROCEEDING
# ============================================================================

first_stage <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid + protestor_violence +
    n_police_stations_5km,
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
  selection = police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid + protestor_violence +
    n_police_stations_5km,          # exclusion restriction
  
  outcome = arrest ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid + protestor_violence,
  
  data   = acled_data,
  method = "2step"
)

summary(heck_arrest)

# ============================================================================
# HECKMAN 2: BRUTALITY
# ============================================================================

cat("\n=== HECKMAN MODEL 2: BRUTALITY ===\n")

heck_brutality <- heckit(
  selection = police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid + protestor_violence +
    n_police_stations_5km,          # exclusion restriction
  
  outcome = brutality ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid + protestor_violence,
  
  data   = acled_data,
  method = "2step"
)

summary(heck_brutality)

# ============================================================================

# Think I have some listwise deletion? 

# Which variables have NAs?
colSums(is.na(acled_data[, all.vars(sel_formula)]))

# Does it add up?
nrow(acled_data) - nrow(na.omit(acled_data[, all.vars(sel_formula)])) # def listwise deletion
# due to some missing vals in the geospatial columns 

# Create a missingness indicator
acled_data$missing_spatial <- is.na(acled_data$log_dist_govt_building) | 
  is.na(acled_data$urb_rur) |
  is.na(acled_data$crowd_size_cat)

# Are dropped cases more/less likely to involve police?
table(acled_data$missing_spatial, acled_data$police_presence_bin)
prop.table(table(acled_data$missing_spatial, acled_data$police_presence_bin), margin = 1)

# 1,490 observations were excluded due to missing spatial covariates. Excluded cases 
# showed lower rates of police presence (0.54% vs 2.63%), consistent with geocoding
# failures disproportionately affecting events in locations with lower state 
# infrastructure visibility. Eight police-presence events were lost. 

# ============================================================================
# IMR DIAGNOSTICS
# ============================================================================

# Inverse Mills Ratio significance tests whether Heckman correction is needed.
# Non-significant IMR = selection bias not a material concern for that outcome.

cat("\n=== IMR DIAGNOSTICS ===\n")
cat("Arrest   — rho:", round(heck_arrest$rho,    4),
    " | IMR p-value:",
    round(summary(heck_arrest)$estimate[
      "invMillsRatio", "Pr(>|t|)"], 4), "\n")

cat("Brutality — rho:", round(heck_brutality$rho, 4),
    " | IMR p-value:",
    round(summary(heck_brutality)$estimate[
      "invMillsRatio", "Pr(>|t|)"], 4), "\n")

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

boot_se_arrest       <- apply(boot_arrest$t,  2, sd, na.rm = TRUE)
names(boot_se_arrest) <- names(coef(heck_arrest))
coefs_arrest          <- coef(heck_arrest)
z_arrest              <- coefs_arrest / boot_se_arrest
p_arrest              <- 2 * pnorm(-abs(z_arrest))

cat("\n=== ARREST: Full results with bootstrap SEs ===\n")
arrest_results <- data.frame(
  Estimate  = round(coefs_arrest,    4),
  Std_Error = round(boot_se_arrest,  4),
  z_value   = round(z_arrest,        4),
  p_value   = round(p_arrest,        4)
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

boot_se_brutality       <- apply(boot_brutality$t, 2, sd, na.rm = TRUE)
names(boot_se_brutality) <- names(coef(heck_brutality))
coefs_brutality          <- coef(heck_brutality)
z_brutality              <- coefs_brutality / boot_se_brutality
p_brutality              <- 2 * pnorm(-abs(z_brutality))

cat("\n=== BRUTALITY: Full results with bootstrap SEs ===\n")
brutality_results <- data.frame(
  Estimate  = round(coefs_brutality,    4),
  Std_Error = round(boot_se_brutality,  4),
  z_value   = round(z_brutality,        4),
  p_value   = round(p_brutality,        4)
)
print(brutality_results)

# ============================================================================
# PARTISAN COEFFICIENTS: OUTCOME EQUATIONS SUMMARY
# ============================================================================
names(coefs_arrest)

outcome_idx <- which(names(coefs_arrest) == "left_pure")[2] + 0:3

heck_comparison <- bind_rows(
  data.frame(
    Variable = c("Left", "Right", "Unknown", "Counter-Protest"),
    Outcome  = "Arrest",
    Estimate = round(coefs_arrest[outcome_idx],    4),
    SE       = round(boot_se_arrest[outcome_idx],  4),
    p_value  = round(p_arrest[outcome_idx],        4)
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
# SAVE RESULTS
# ============================================================================

write.csv(heck_comparison,   "results/heckman_partisan_coefs.csv",    row.names = FALSE)
write.csv(arrest_results,    "results/heckman_arrest_full.csv",        row.names = FALSE)
write.csv(brutality_results, "results/heckman_brutality_full.csv",     row.names = FALSE)
saveRDS(heck_arrest,         "models/heckman_arrest.rds")
saveRDS(heck_brutality,      "models/heckman_brutality.rds")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to results/ and models/ directories\n")

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
# Choose delta: anchor to a fraction of the Model 1 left-right deployment
# gap (0.414 log-odds). Using half that as the equivalence bound is a
# defensible a priori choice — an effect at the conduct stage smaller than
# half the magnitude of the deployment-stage gap is treated as practically
# negligible for the paper's theoretical purposes. Adjust as needed.
# ----------------------------------------------------------------------

delta_primary <- 0.18355

# Also report a design-based / achieved bound: the smallest delta for
# which this SE would have ~80% power to detect equivalence, so readers
# can see what the n=155 right-coded subsample can and can't rule out.
# Rule of thumb for 80% power TOST: delta_80 ≈ SE * (1.645 + 0.84) ≈ SE * 2.49
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

# Save
tost_results <- bind_rows(
  cbind(Outcome = "Arrest", tost_arrest),
  cbind(Outcome = "Brutality", tost_brutality)
)
write.csv(tost_results, "results/tost_equivalence.csv", row.names = FALSE)

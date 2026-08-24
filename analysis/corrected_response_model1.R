# Logit Model 1: Police Presence — Baseline, Aligned to Corrected Model 2
# Author: Katie Nutley
# Date: 20-08-2026
#
# This is the baseline specification for Table 8 / the manuscript's Model 1.
# It is built to contain EXACTLY the same right-hand-side terms as Model 2's
# corrected sel_formula (02_heckman_severity_CORRECTED.R), MINUS:
#   - n_police_stations_5km   (the exclusion restriction; selection-equation
#                               only, has no place in a standalone Model 1)
#   - protestor_violence      (deliberately withheld from Model 1's baseline
#                               by design — it enters as a separate robustness
#                               step, model_violence, exactly as in the
#                               original script; Model 2 already had violence
#                               in its baseline, so it stays there)
#
# Do NOT add protest_goal or year_c here — those are separate exploratory
# additions from a different comparison script and are not part of this
# correction. Adding them would be a new modeling decision requiring its own
# justification in Section 6.1, not a bug fix.

# ============================================================================
# PACKAGES
# ============================================================================

library(tidyverse)
library(dplyr)
library(sandwich)
library(lmtest)
library(stargazer)
library(car)
library(ggplot2)

# ============================================================================
# DATA PREPARATION
# ============================================================================

acled_data <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

acled_data <- acled_data %>%
  mutate(
    crowd_size_cat = relevel(factor(crowd_size_cat), ref = "No report"),
    urb_rur        = relevel(factor(urb_rur),        ref = "Predominantly urban"),
    
    log_dist_govt_building  = log1p(dist_govt_building_m),
    log_dist_major_road     = log1p(dist_major_road_m),
    log_dist_police_station = log1p(dist_police_station_m)
  )

cat("is_weekend present:", "is_weekend" %in% names(acled_data), "\n")
cat("log_dist_major_road missing:", sum(is.na(acled_data$log_dist_major_road)), "\n")
cat("is_weekend missing:", sum(is.na(acled_data$is_weekend)), "\n")

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
    covid        = as.integer(year %in% c(2020, 2021))
  )

table(acled_data$counter_protest)
table(acled_data$covid)
table(acled_data$is_weekend)

# ============================================================================
# DESCRIPTIVE: POLICE PRESENCE RATE BY PARTISAN TYPE
# ============================================================================

acled_data %>%
  group_by(event_partisan_type_final) %>%
  summarise(
    n_events   = n(),
    n_police   = sum(police_presence_bin, na.rm = TRUE),
    pct_police = mean(police_presence_bin, na.rm = TRUE) * 100
  ) %>%
  arrange(desc(pct_police))

# ============================================================================
# MODEL 1 (BASELINE): SAME CONTROL SET AS MODEL 2's sel_formula,
# minus n_police_stations_5km (exclusion restriction) and protestor_violence
# ============================================================================

model_collapsed <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country +
    crowd_size_cat +
    log_dist_govt_building +
    log_dist_major_road +
    urb_rur +
    covid +
    is_weekend,
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_collapsed <- vcovHC(model_collapsed, type = "HC3")

cat("\n=== MODEL 1 (BASELINE, ALIGNED TO MODEL 2) ===\n")
coeftest(model_collapsed, vcov = vcov_collapsed)

cat("\nObservations used:", nrow(model_collapsed$model), "\n")
cat("Observations dropped (listwise deletion):",
    nrow(acled_data) - nrow(model_collapsed$model), "\n")
# Compare this dropped-N against Model 2's "Observations dropped due to
# listwise deletion" figure. They will NOT be identical — Model 2 also drops
# on n_police_stations_5km and protestor_violence, which aren't in this
# model — but they should be close, and Model 1's N should be >= Model 2's N.

# ============================================================================
# MODEL 2 (ROBUSTNESS): + PROTESTOR VIOLENCE
# ============================================================================
# This now matches Model 2 (Heckman)'s sel_formula exactly, minus only the
# exclusion restriction itself.

model_violence <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country +
    crowd_size_cat +
    log_dist_govt_building +
    log_dist_major_road +
    urb_rur +
    covid +
    is_weekend +
    protestor_violence,
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_violence <- vcovHC(model_violence, type = "HC3")
cat("\n=== MODEL 2 (+ Protestor Violence; matches sel_formula minus exclusion) ===\n")
coeftest(model_violence, vcov = vcov_violence)

# ============================================================================
# MODEL 3 (ROBUSTNESS): + DISTANCE TO POLICE STATION
# ============================================================================
# Verifies sign/magnitude of the planned exclusion restriction as a control,
# before it's used as an actual exclusion restriction in the Heckman model.

model_police_dist <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country +
    crowd_size_cat +
    log_dist_govt_building +
    log_dist_major_road +
    urb_rur +
    covid +
    is_weekend +
    protestor_violence +
    log_dist_police_station,
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_police_dist <- vcovHC(model_police_dist, type = "HC3")
cat("\n=== MODEL 3 (+ Police Station Distance) ===\n")
coeftest(model_police_dist, vcov = vcov_police_dist)

cat("\nObservations lost to Canary Islands NA on police station distance:\n")
cat(nrow(model_violence$model) - nrow(model_police_dist$model), "\n")

# ============================================================================
# DIAGNOSTICS
# ============================================================================

cat("\n--- Model 1: Baseline ---\n")
cat("Null deviance:    ", model_collapsed$null.deviance, "\n")
cat("Residual deviance:", model_collapsed$deviance, "\n")
cat("AIC:              ", AIC(model_collapsed), "\n")

cat("\n--- Model 2: + Violence ---\n")
cat("Residual deviance:", model_violence$deviance, "\n")
cat("AIC:              ", AIC(model_violence), "\n")

cat("\n--- Model 3: + Police station distance ---\n")
cat("Residual deviance:", model_police_dist$deviance, "\n")
cat("AIC:              ", AIC(model_police_dist), "\n")

# ============================================================================
# ROBUSTNESS: PARTISAN COEFFICIENTS ACROSS SPECIFICATIONS
# ============================================================================

partisan_vars <- c("left_pure", "right_pure", "unknown_pure", "counter_protest")

extract_partisan <- function(model, vcov, label) {
  coefs <- coef(model)[partisan_vars]
  ses   <- sqrt(diag(vcov))[partisan_vars]
  data.frame(
    Variable = partisan_vars,
    Model    = label,
    Estimate = round(coefs, 4),
    SE       = round(ses,   4),
    p_value  = round(2 * pnorm(-abs(coefs / ses)), 4)
  )
}

comparison <- bind_rows(
  extract_partisan(model_collapsed,   vcov_collapsed,   "Baseline (aligned to Model 2)"),
  extract_partisan(model_violence,    vcov_violence,    "+ Protestor violence"),
  extract_partisan(model_police_dist, vcov_police_dist, "+ Police station dist")
)

cat("\n=== PARTISAN COEFFICIENTS ACROSS SPECIFICATIONS ===\n")
print(comparison, row.names = FALSE)

# ============================================================================
# CROSS-CHECK AGAINST MODEL 2'S SELECTION EQUATION
# ============================================================================
# model_violence's partisan coefficients should be numerically IDENTICAL (up
# to estimation method — this is a standalone logit, Model 2's first stage is
# a probit) in SIGN and roughly comparable in SIGNIFICANCE PATTERN to the
# selection-equation block of heck_arrest / heck_brutality in
# 02_heckman_severity_CORRECTED.R, since the RHS is now the same minus the
# exclusion restriction. If signs or significance flip between this and the
# Heckman first stage, something is misaligned — investigate before trusting
# either table.

cat("\n=== SANITY CHECK: compare partisan rows above against Model 2's ===\n")
cat("=== probit selection-equation output (left_pure, right_pure, etc.)  ===\n")
cat("Signs and significance pattern should match; magnitudes will differ\n")
cat("due to logit vs. probit link functions.\n")

# ============================================================================
# ODDS RATIOS AND CONFIDENCE INTERVALS (MAIN MODEL — for Table 8)
# ============================================================================

coefs        <- coef(model_collapsed)[partisan_vars]
odds_ratios  <- exp(coefs)
se_robust    <- sqrt(diag(vcov_collapsed))[partisan_vars]
ci_lower     <- exp(coefs - 1.96 * se_robust)
ci_upper     <- exp(coefs + 1.96 * se_robust)
z_stats      <- coefs / se_robust
p_values     <- 2 * pnorm(-abs(z_stats))

results_or <- data.frame(
  Variable    = c("Left", "Right", "Unknown", "Counter-Protest"),
  Coefficient = coefs,
  Std_Error   = se_robust,
  Odds_Ratio  = odds_ratios,
  CI_Lower    = ci_lower,
  CI_Upper    = ci_upper,
  P_Value     = p_values,
  Significant = p_values < 0.05
) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n=== TABLE 8 SOURCE: ODDS RATIOS (BASELINE MODEL) ===\n")
print(results_or)

# ============================================================================
# HYPOTHESIS TESTS
# ============================================================================

cat("\n=== HYPOTHESIS TESTS ===\n\n")

cat("H1: Left vs Right protests\n")
h1_test <- linearHypothesis(model_collapsed, "left_pure = right_pure", vcov = vcov_collapsed)
print(h1_test)

cat("\nH2: Left > Right (one-tailed)\n")
diff_lr    <- coef(model_collapsed)["left_pure"] - coef(model_collapsed)["right_pure"]
se_diff_lr <- sqrt(
  vcov_collapsed["left_pure",  "left_pure"]  +
    vcov_collapsed["right_pure", "right_pure"] -
    2 * vcov_collapsed["left_pure", "right_pure"]
)
z_lr         <- diff_lr / se_diff_lr
p_lr_onetail <- pnorm(z_lr, lower.tail = FALSE)
cat("Coefficient difference:", round(diff_lr, 4),
    "\nZ-statistic:",          round(z_lr, 3),
    "\nOne-tailed p-value:",   round(p_lr_onetail, 4), "\n")

cat("\nH3: Extremism hypothesis (Left and Right > Centre)\n")
z_left  <- coef(model_collapsed)["left_pure"]  / sqrt(diag(vcov_collapsed))["left_pure"]
p_left  <- pnorm(z_left,  lower.tail = FALSE)
cat("Left > Centre:  z =", round(z_left,  3), ", one-tailed p =", round(p_left,  4), "\n")

z_right <- coef(model_collapsed)["right_pure"] / sqrt(diag(vcov_collapsed))["right_pure"]
p_right <- pnorm(z_right, lower.tail = FALSE)
cat("Right > Centre: z =", round(z_right, 3), ", one-tailed p =", round(p_right, 4), "\n")

cat("\nH4: Counter-protests vs single-partisan events\n")
h4_test <- linearHypothesis(
  model_collapsed,
  c("counter_protest = left_pure", "counter_protest = right_pure"),
  vcov = vcov_collapsed
)
print(h4_test)

# ============================================================================
# PREDICTED PROBABILITIES BY PARTISAN TYPE
# ============================================================================

modal_country       <- names(sort(table(acled_data$country),        decreasing = TRUE)[1])
modal_crowd_size    <- names(sort(table(acled_data$crowd_size_cat), decreasing = TRUE)[1])
modal_urb_rur       <- "Predominantly urban"
modal_log_dist_govt <- median(acled_data$log_dist_govt_building, na.rm = TRUE)
modal_log_dist_road <- median(acled_data$log_dist_major_road,    na.rm = TRUE)
modal_is_weekend    <- as.integer(names(sort(table(acled_data$is_weekend), decreasing = TRUE)[1]))

pred_data <- data.frame(
  left_pure              = c(1, 0, 0, 0, 0, 0),
  right_pure             = c(0, 1, 0, 0, 0, 0),
  unknown_pure           = c(0, 0, 1, 0, 0, 0),
  counter_protest        = c(0, 0, 0, 0, 1, 0),
  country                = rep(modal_country, 6),
  crowd_size_cat         = rep(modal_crowd_size, 6),
  log_dist_govt_building = rep(modal_log_dist_govt, 6),
  log_dist_major_road    = rep(modal_log_dist_road, 6),
  urb_rur                = rep(modal_urb_rur, 6),
  covid                  = rep(0L, 6),
  is_weekend             = rep(modal_is_weekend, 6)
)

pred_probs <- predict(model_collapsed, newdata = pred_data, type = "response")

pred_summary <- data.frame(
  Partisan_Type  = c("Left", "Right", "Unknown", "Counter-Protest",
                     "Centre (Reference)", ""),
  Predicted_Prob = round(pred_probs * 100, 2),
  N_Events       = c(
    sum(acled_data$left_pure,       na.rm = TRUE),
    sum(acled_data$right_pure,      na.rm = TRUE),
    sum(acled_data$unknown_pure,    na.rm = TRUE),
    sum(acled_data$counter_protest, na.rm = TRUE),
    sum(acled_data$event_partisan_type_final == "centre", na.rm = TRUE),
    NA
  )
) %>% filter(Partisan_Type != "")

cat("\n=== PREDICTED PROBABILITIES ===\n")
print(pred_summary)

# ============================================================================
# STANDARD ERROR CLUSTERING (ADMIN2) — FOR TABLE 13
# ============================================================================

n_distinct(acled_data$admin2)
acled_clean <- acled_data %>% filter(!is.na(admin2))

model_collapsed_clean <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    log_dist_major_road + urb_rur + covid + is_weekend,
  data   = acled_clean,
  family = binomial(link = "logit")
)

vcov_admin2 <- vcovCL(model_collapsed_clean, cluster = ~admin2)

cat("\n=== ADMIN2-CLUSTERED (TABLE 13) ===\n")
coeftest(model_collapsed_clean, vcov = vcov_admin2)

cat("\nN (admin2-clean sample):", nrow(model_collapsed_clean$model), "\n")
cat("Dropped for missing admin2:", 109151 - nrow(model_collapsed_clean$model), "\n")

vcov_collapsed_clean <- vcovHC(model_collapsed_clean, type = "HC3")
table13_vars <- c("left_pure", "right_pure", "unknown_pure", "counter_protest")

table13 <- data.frame(
  Variable   = c("Left", "Right", "Unknown", "Counter-Protest"),
  Est_HC3    = round(coef(model_collapsed_clean)[table13_vars], 4),
  SE_HC3     = round(sqrt(diag(vcov_collapsed_clean))[table13_vars], 4),
  Est_Clust  = round(coef(model_collapsed_clean)[table13_vars], 4),
  SE_Clust   = round(sqrt(diag(vcov_admin2))[table13_vars], 4)
)
cat("\n=== TABLE 13 (HC3 vs. admin2-clustered, same spec) ===\n")
print(table13, row.names = FALSE)

# ============================================================================
# SAVE
# ============================================================================

write.csv(results_or,    "results/table8_odds_ratios.csv",              row.names = FALSE)
write.csv(comparison,    "results/model1_partisan_coefs_across_specs.csv", row.names = FALSE)
write.csv(pred_summary,  "results/predicted_probabilities.csv",         row.names = FALSE)
write.csv(table13,       "results/table13_admin2_clustered.csv",        row.names = FALSE)
saveRDS(model_collapsed,       "models/model1_baseline_aligned.rds")
saveRDS(model_violence,        "models/model1_violence.rds")
saveRDS(model_police_dist,     "models/model1_police_dist.rds")
saveRDS(model_collapsed_clean, "models/model1_admin2_clean.rds")

cat("\n=== Analysis Complete ===\n")
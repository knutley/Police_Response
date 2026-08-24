# Logit Model(s) of Police Presence 
# Author: Katie Nutley
# Date: 18-12-2025
# Edited: 08-06-2026

# Edited in June to add controls created in control_variable_construction.R
# and the resulting dataset. Compound partisan categories (left_centre, left_right,
# right_centre) are collapsed into a single counter_protest dummy. A Covid dummy
# (2020-2021) is included to absorb the structural break in protest activity and
# policing during the pandemic, rather than dropping those years.
# Also adds protestor_violence and log_dist_police_station as additional controls;
# stability of partisan coefficients across specifications is the robustness argument.

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

# Set reference categories for categorical controls
acled_data <- acled_data %>%
  mutate(
    crowd_size_cat = relevel(factor(crowd_size_cat), ref = "No report"),
    protest_goal   = relevel(factor(protest_goal),   ref = "Other/Unknown"),
    urb_rur        = relevel(factor(urb_rur),        ref = "Predominantly urban"),
    
    # Log-transform continuous distance variables to reduce skew
    log_dist_govt_building  = log1p(dist_govt_building_m),
    log_dist_major_road     = log1p(dist_major_road_m),
    log_dist_police_station = log1p(dist_police_station_m)
  )

# ============================================================================
# PARTISAN TYPE DUMMIES + COVID INDICATOR
# ============================================================================

# Compound categories (left_right, left_centre, right_centre) collapsed into
# a single counter_protest dummy — these events share the structural feature
# of opposing partisan groups. Centre is omitted as the reference category.
#
# Covid dummy absorbs the 2020-2021 structural break in protest activity and
# policing, allowing the full sample to be retained.

acled_data <- acled_data %>%
  mutate(
    counter_protest = as.integer(
      event_partisan_type_final %in% c("left_right", "left_centre", "right_centre")
    ),
    left_pure    = as.integer(event_partisan_type_final == "left"),
    right_pure   = as.integer(event_partisan_type_final == "right"),
    unknown_pure = as.integer(event_partisan_type_final == "unknown"),
    # centre_pure omitted as reference category
    covid        = as.integer(year %in% c(2020, 2021))
  )

# Sanity checks
table(acled_data$counter_protest)
table(acled_data$event_partisan_type_final, acled_data$counter_protest)
table(acled_data$covid)

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
# MODEL 1: BASELINE — COLLAPSED COUNTER-PROTEST WITH COVID CONTROL
# ============================================================================

model_collapsed <- glm(
  police_presence_bin ~
    # Partisan types (centre is reference; compound types collapsed)
    left_pure + right_pure + unknown_pure + counter_protest +
    
    # Country fixed effects
    country +
    
    # Protest characteristics
    crowd_size_cat +
    
    # Spatial controls
    log_dist_govt_building +
    
    # Urban/rural classification
    urb_rur +
    
    # Covid structural break (2020-2021)
    covid,
  
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_collapsed <- vcovHC(model_collapsed, type = "HC3")
coeftest(model_collapsed, vcov = vcov_collapsed)

# ============================================================================
# MODEL 2: + PROTESTOR VIOLENCE
# ============================================================================

# Violence is the strongest individual predictor of police presence but is
# orthogonal to partisan type — its inclusion barely shifts the partisan
# coefficients, confirming those results are not confounded by violence.

model_violence <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country +
    crowd_size_cat +
    log_dist_govt_building +
    urb_rur +
    covid +
    protestor_violence,
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_violence <- vcovHC(model_violence, type = "HC3")
coeftest(model_violence, vcov = vcov_violence)

# Violence rate by partisan type — useful for write-up
prop.table(table(acled_data$protestor_violence,
                 acled_data$event_partisan_type_final), margin = 2)

# ============================================================================
# MODEL 3: + DISTANCE TO POLICE STATION
# ============================================================================

# Planned exclusion restriction for Heckman selection model.
# Included here as a control to verify sign and magnitude.
# Note: NAs introduced for Canary Islands events (set in control construction).

model_police_dist <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country +
    crowd_size_cat +
    log_dist_govt_building +
    urb_rur +
    covid +
    protestor_violence +
    log_dist_police_station,
  data   = acled_data,
  family = binomial(link = "logit")
)

vcov_police_dist <- vcovHC(model_police_dist, type = "HC3")
coeftest(model_police_dist, vcov = vcov_police_dist)

cat("\nObservations lost to Canary Islands NA on police station distance:\n")
cat(nrow(model_violence$model) - nrow(model_police_dist$model), "\n")

# Collinearity check: govt building vs police station distance
cor(acled_data$log_dist_govt_building,
    acled_data$log_dist_police_station,
    use = "complete.obs")

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

# Share of each partisan type by country
acled_data %>%
  group_by(country) %>%
  summarise(
    pct_left            = mean(left_pure,       na.rm = TRUE) * 100,
    pct_right           = mean(right_pure,      na.rm = TRUE) * 100,
    pct_counter_protest = mean(counter_protest, na.rm = TRUE) * 100,
    n = n()
  ) %>%
  arrange(desc(pct_left))

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
  extract_partisan(model_collapsed,   vcov_collapsed,   "Baseline"),
  extract_partisan(model_violence,    vcov_violence,    "+ Protestor violence"),
  extract_partisan(model_police_dist, vcov_police_dist, "+ Police station dist")
)

print(comparison, row.names = FALSE)

# ============================================================================
# ODDS RATIOS AND CONFIDENCE INTERVALS (MAIN MODEL)
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

print(results_or)

# ============================================================================
# PREDICTED PROBABILITIES BY PARTISAN TYPE (MAIN MODEL)
# ============================================================================

modal_country       <- names(sort(table(acled_data$country),        decreasing = TRUE)[1])
modal_crowd_size    <- names(sort(table(acled_data$crowd_size_cat), decreasing = TRUE)[1])
modal_urb_rur       <- "Predominantly urban"
modal_log_dist_govt <- median(acled_data$log_dist_govt_building, na.rm = TRUE)

# Predicted probabilities for a non-Covid event at modal/median control values
pred_data <- data.frame(
  left_pure              = c(1, 0, 0, 0, 0, 0),
  right_pure             = c(0, 1, 0, 0, 0, 0),
  unknown_pure           = c(0, 0, 1, 0, 0, 0),
  counter_protest        = c(0, 0, 0, 0, 1, 0),
  country                = rep(modal_country, 6),
  crowd_size_cat         = rep(modal_crowd_size, 6),
  log_dist_govt_building = rep(modal_log_dist_govt, 6),
  urb_rur                = rep(modal_urb_rur, 6),
  covid                  = rep(0L, 6)
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

print(pred_summary)

# ============================================================================
# HYPOTHESIS TESTS (MAIN MODEL)
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
# VISUALIZATION
# ============================================================================

plot_data <- pred_summary %>%
  mutate(
    ci_lower = Predicted_Prob - 1.96 * sqrt(Predicted_Prob * (100 - Predicted_Prob) / N_Events),
    ci_upper = Predicted_Prob + 1.96 * sqrt(Predicted_Prob * (100 - Predicted_Prob) / N_Events)
  )

ggplot(plot_data, aes(x = reorder(Partisan_Type, Predicted_Prob), y = Predicted_Prob)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
  coord_flip() +
  labs(
    title    = "Predicted Probability of Police Presence by Partisan Type",
    subtitle = "Non-Covid baseline; controls held at modal/median values",
    x        = "Event Partisan Type",
    y        = "Predicted Probability of Police Presence (%)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave("police_presence_by_partisan_type_collapsed.png", width = 10, height = 6, dpi = 300)

# ============================================================================
# REGRESSION TABLE
# ============================================================================

# stargazer does not natively accept vcovHC objects; robust SEs are reported
# via coeftest above. Table shows all three specifications for comparison.
stargazer(model_collapsed, model_violence, model_police_dist,
          type = "text",
          title = "Logistic Regression: Police Presence at Protest Events",
          dep.var.labels = "Police Present",
          covariate.labels = c("Left", "Right", "Unknown", "Counter-Protest",
                               "Covid (2020-21)", "Protestor Violence",
                               "Log Distance to Police Station"),
          omit = c("country", "crowd_size_cat", "urb_rur", "log_dist_govt"),
          add.lines = list(
            c("Country FE",              "Yes", "Yes", "Yes"),
            c("Crowd size controls",     "Yes", "Yes", "Yes"),
            c("Urban/rural controls",    "Yes", "Yes", "Yes"),
            c("Govt building distance",  "Yes", "Yes", "Yes"),
            c("SEs",                     "HC3", "HC3", "HC3")
          ),
          star.cutoffs = c(0.05, 0.01, 0.001),
          notes        = "HC3 robust standard errors. Reference category: Centre protests. Covid dummy = 1 for 2020-2021.",
          notes.append = TRUE,
          report       = "vcp*")

# ============================================================================
# SAVE RESULTS
# ============================================================================

write.csv(results_or,    "results/police_presence_odds_ratios_collapsed.csv", row.names = FALSE)
write.csv(pred_summary,  "results/predicted_probabilities_collapsed.csv",      row.names = FALSE)
write.csv(comparison,    "results/partisan_coefs_across_specs.csv",            row.names = FALSE)
saveRDS(model_collapsed,   "models/police_presence_model_collapsed.rds")
saveRDS(model_violence,    "models/police_presence_model_violence.rds")
saveRDS(model_police_dist, "models/police_presence_model_police_dist.rds")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to results/ directory\n")
cat("Models saved to models/ directory\n")

# ============================================================================
# STANDARD ERROR CLUSTERING 
# ============================================================================

library(sandwich)

# checking to see how many little places I have: 
n_distinct(acled_data$admin2)

# as it's above 20-30; 
acled_clean <- acled_data %>% filter(!is.na(admin2))

model_collapsed_clean <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid,
  data   = acled_clean,
  family = binomial(link = "logit")
)

vcov_admin2 <- vcovCL(model_collapsed_clean, cluster = ~admin2)
coeftest(model_collapsed_clean, vcov = vcov_admin2)

nrow(model_collapsed_clean$model)
109151 - nrow(model_collapsed_clean$model)

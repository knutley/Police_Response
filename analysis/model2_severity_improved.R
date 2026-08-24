# ============================================================================
# Model 1 (Police Presence): Baseline vs. Expanded Control Specification
# Compares the model as currently reported in the paper against a version
# that adds protest goals, road proximity, weekend indicator, and year trend
# ============================================================================

library(tidyverse)
library(sandwich)
library(lmtest)
library(car)

acled_data <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

# ----------------------------------------------------------------------------
# Prep: dummies, reference levels, transforms (mirrors your existing pipeline)
# ----------------------------------------------------------------------------

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
    urb_rur        = relevel(factor(urb_rur), ref = "Predominantly urban"),
    log_dist_govt_building = log1p(dist_govt_building_m),
    
    # newly added controls
    log_dist_major_road = log1p(dist_major_road_m),
    protest_goal    = relevel(factor(protest_goal), ref = "Other/Unknown"),
    is_weekend      = as.integer(is_weekend),
    year_c          = year - min(year, na.rm = TRUE)  # centred year trend
  )

# ----------------------------------------------------------------------------
# BASELINE MODEL — as currently reported in the paper (Model 1 / Table 8)
# ----------------------------------------------------------------------------

baseline <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid,
  data   = acled_data,
  family = binomial(link = "logit")
)

cat("\n=== BASELINE MODEL ===\n")
coeftest(baseline, vcov = vcovHC(baseline, type = "HC3"))

# ----------------------------------------------------------------------------
# EXPANDED MODEL — adds protest goals, road proximity, weekend, year trend
# ----------------------------------------------------------------------------

expanded <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid +
    protest_goal + log_dist_major_road + is_weekend + year_c,
  data   = acled_data,
  family = binomial(link = "logit")
)

cat("\n=== EXPANDED MODEL (with previously-unused controls) ===\n")
coeftest(expanded, vcov = vcovHC(expanded, type = "HC3"))

# ----------------------------------------------------------------------------
# COMPARE: do the partisan coefficients move?
# ----------------------------------------------------------------------------

extract_partisan <- function(model, label) {
  ct <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  vars <- c("left_pure", "right_pure", "unknown_pure", "counter_protest")
  data.frame(
    model    = label,
    variable = vars,
    estimate = ct[vars, "Estimate"],
    se       = ct[vars, "Std. Error"],
    p_value  = ct[vars, "Pr(>|z|)"]
  )
}

comparison <- bind_rows(
  extract_partisan(baseline, "Baseline"),
  extract_partisan(expanded, "Expanded")
)

cat("\n=== PARTISAN COEFFICIENT COMPARISON ===\n")
comparison %>%
  pivot_wider(names_from = model, values_from = c(estimate, se, p_value)) %>%
  mutate(across(where(is.numeric), ~round(., 4))) %>%
  print(row.names = FALSE)

# ----------------------------------------------------------------------------
# Key theoretical contrast: left vs. right (H1/H2 depending on your numbering)
# ----------------------------------------------------------------------------

cat("\n=== LEFT vs. RIGHT CONTRAST ===\n")
cat("Baseline:\n")
print(linearHypothesis(baseline, "left_pure = right_pure",
                       vcov. = vcovHC(baseline, type = "HC3")))

cat("\nExpanded:\n")
print(linearHypothesis(expanded, "left_pure = right_pure",
                       vcov. = vcovHC(expanded, type = "HC3")))

# ----------------------------------------------------------------------------
# Does the expanded model actually fit better? (LR test, note: not HC3-robust)
# ----------------------------------------------------------------------------

cat("\n=== LIKELIHOOD RATIO TEST (nested models, same N required) ===\n")
# NB: only valid if baseline and expanded have identical listwise-deleted N.
# Check this before trusting the test:
cat("Baseline N:", nobs(baseline), " | Expanded N:", nobs(expanded), "\n")

if (nobs(baseline) == nobs(expanded)) {
  print(anova(baseline, expanded, test = "LRT"))
} else {
  cat("N differs — likely due to missingness in the new controls (e.g. road distance).\n")
  cat("Refit baseline on the expanded model's estimation sample before comparing.\n")
}

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------

write.csv(comparison, "results/model1_baseline_vs_expanded.csv", row.names = FALSE)
saveRDS(baseline, "models/model1_baseline.rds")
saveRDS(expanded, "models/model1_expanded.rds")

# ============================================================================
# Model 1 (Police Presence): Baseline vs. Expanded Control Specification
# v2 — protest_goal removed (mediator concern); retains road proximity,
# weekend indicator, and year trend
# ============================================================================

library(tidyverse)
library(sandwich)
library(lmtest)
library(car)

acled_data <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv")

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
    urb_rur        = relevel(factor(urb_rur), ref = "Predominantly urban"),
    log_dist_govt_building = log1p(dist_govt_building_m),
    
    log_dist_major_road = log1p(dist_major_road_m),
    is_weekend      = as.integer(is_weekend),
    year_c          = year - min(year, na.rm = TRUE)
  )

# ----------------------------------------------------------------------------
# BASELINE MODEL — as currently reported in the paper
# ----------------------------------------------------------------------------

baseline <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid,
  data   = acled_data,
  family = binomial(link = "logit")
)

cat("\n=== BASELINE MODEL ===\n")
coeftest(baseline, vcov = vcovHC(baseline, type = "HC3"))

# ----------------------------------------------------------------------------
# EXPANDED MODEL — road proximity, weekend, year trend (no protest_goal)
# ----------------------------------------------------------------------------

expanded <- glm(
  police_presence_bin ~
    left_pure + right_pure + unknown_pure + counter_protest +
    country + crowd_size_cat + log_dist_govt_building +
    urb_rur + covid +
    log_dist_major_road + is_weekend + year_c,
  data   = acled_data,
  family = binomial(link = "logit")
)

cat("\n=== EXPANDED MODEL (road proximity + weekend + year trend) ===\n")
coeftest(expanded, vcov = vcovHC(expanded, type = "HC3"))

# ----------------------------------------------------------------------------
# COMPARE
# ----------------------------------------------------------------------------

extract_partisan <- function(model, label) {
  ct <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  vars <- c("left_pure", "right_pure", "unknown_pure", "counter_protest")
  data.frame(
    model    = label,
    variable = vars,
    estimate = ct[vars, "Estimate"],
    se       = ct[vars, "Std. Error"],
    p_value  = ct[vars, "Pr(>|z|)"]
  )
}

comparison <- bind_rows(
  extract_partisan(baseline, "Baseline"),
  extract_partisan(expanded, "Expanded")
)

cat("\n=== PARTISAN COEFFICIENT COMPARISON ===\n")
comparison %>%
  pivot_wider(names_from = model, values_from = c(estimate, se, p_value)) %>%
  mutate(across(where(is.numeric), ~round(., 4))) %>%
  print(row.names = FALSE)

# ----------------------------------------------------------------------------
# Left vs. right contrast
# ----------------------------------------------------------------------------

cat("\n=== LEFT vs. RIGHT CONTRAST ===\n")
cat("Baseline:\n")
print(linearHypothesis(baseline, "left_pure = right_pure",
                       vcov. = vcovHC(baseline, type = "HC3")))

cat("\nExpanded:\n")
print(linearHypothesis(expanded, "left_pure = right_pure",
                       vcov. = vcovHC(expanded, type = "HC3")))

# ----------------------------------------------------------------------------
# LR test (check N matches first)
# ----------------------------------------------------------------------------

cat("\n=== LIKELIHOOD RATIO TEST ===\n")
cat("Baseline N:", nobs(baseline), " | Expanded N:", nobs(expanded), "\n")

if (nobs(baseline) == nobs(expanded)) {
  print(anova(baseline, expanded, test = "LRT"))
} else {
  cat("N differs — refit baseline on expanded's estimation sample before comparing.\n")
}

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------

write.csv(comparison, "results/model1_baseline_vs_expanded_v2.csv", row.names = FALSE)
saveRDS(baseline, "models/model1_baseline.rds")
saveRDS(expanded, "models/model1_expanded_v2.rds")

# Diagnostic: is crowd size predicted by partisan type?
# (Informs whether it's a plausible mediator, not a tool for improving p-values)

crowd_by_partisan <- acled_data %>%
  filter(!is.na(crowd_size_cat)) %>%
  group_by(event_partisan_type_final) %>%
  count(crowd_size_cat) %>%
  mutate(pct = n / sum(n) * 100) %>%
  filter(event_partisan_type_final %in% c("left", "right", "centre"))

print(crowd_by_partisan, n = 30)

# Formal test: does partisan type predict crowd size category?
crowd_chisq <- chisq.test(
  table(acled_data$event_partisan_type_final[acled_data$event_partisan_type_final %in% c("left","right","centre")],
        acled_data$crowd_size_cat[acled_data$event_partisan_type_final %in% c("left","right","centre")])
)
print(crowd_chisq)

# Title: Partisanship Classification - Addendum (NLP Bootstrap) v2
# Author: Katelyn Nutley (revised)
# Date: 24-02-2026
# Description: NLP-based bootstrap to classify unknown actors in ITALY ACLED data.
#              For mixed events (classified / unknown assoc_actors co-occurring),
#              uses the notes field alongside the classified actor as a partisan
#              anchor to contextualise unknown actors via zero-shot classification.
#              Requires: italy_acled_partisan_classification.csv from main script.
#
# Changes from v1:
#   - Candidate labels rewritten for Italian protest context
#   - multi_label = TRUE to recover right_and_left / right_and_centre etc.
#   - Combined-label logic replaces forced single-label output
#   - Post-hoc correction block updated to reflect revised label set
#   - Bug fix: random_sample3 now draws from correct object (text_classified_obs2)

library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(reticulate)
library(writexl)

reticulate::py_install("transformers")
reticulate::py_install("torch")

# Load output from main classification script
italy_acled_df <- read_csv("~/Downloads/italy_acled_partisan_classification.csv")

################################################################################
# STEP 1: IDENTIFY MIXED EVENTS
################################################################################

# Logic: if an event has at least one classified actor and at least one unknown
# actor, use NLP on the notes field to contextualise political behaviour.

italy_acled_df <- italy_acled_df %>%
  mutate(
    has_classified = as.integer(
      assoc_actor1_left == 1 | assoc_actor1_right == 1 | assoc_actor1_centre == 1 |
        assoc_actor2_left == 1 | assoc_actor2_right == 1 | assoc_actor2_centre == 1
    ),
    has_unknown = as.integer(
      assoc_actor1_unknown == 1 | assoc_actor2_unknown == 1
    ),
    mixed = as.integer(has_classified == 1 & has_unknown == 1)
  )

cat("Mixed events for NLP classification:", sum(italy_acled_df$mixed == 1), "\n")

################################################################################
# STEP 2: ZERO-SHOT TEXT CLASSIFICATION
################################################################################

# Point to your Python environment - update path as needed
# use_condaenv("your_env_name")
# use_virtualenv("path/to/venv")

transformers <- import("transformers")
torch <- import("torch")

classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli"
)

# Labels revised for Italian protest context, derived from manual coding patterns:
#
# LEFT:   Italian left protests code via labor rights, union organising (esp. CGIL),
#         housing rights, student occupations, precarious work, public service
#         protection, migrant worker rights, and anti-fascist/anti-far-right action.
#
# RIGHT:  Italian right protests code via opposition to government restrictions
#         (coronavirus, green pass), anti-EU agricultural/environmental policy,
#         law and order / security personnel, anti-vaccine mandates, and
#         national sovereignty framing.
#
# CENTRE: Italian centrist protests code via business sector economic grievances,
#         employer/trade associations seeking state relief, professional category
#         survival demands, and civic/constitutional non-partisan framing.

candidate_labels <- c(
  "left-wing protest for causes such as workers rights, labor union organising,
   housing rights for tenants, student movements and school occupations,
   precarious work, public service protection, migrant worker rights,
   or anti-fascist and anti-far-right action",
  "right-wing protest against causes such as government coronavirus restrictions,
   green pass vaccine mandates, EU agricultural or environmental policy,
   in favour of law and order or security personnel, national sovereignty,
   or anti-lockdown business freedom",
  "centrist or non-partisan protest for causes such as business sector economic
   relief, employer or trade association grievances, professional category
   survival, civic safety demands, or road safety and victims rights"
)

# Short names used in label_map and output columns
label_short <- c("left", "right", "centre")

# Threshold for a label to be counted as present (multi-label)
score_threshold <- 0.45

classify_text_multi <- function(text, classifier, labels, threshold, short_names) {
  tryCatch({
    result <- classifier(text, labels, multi_label = TRUE)
    
    scores <- setNames(unlist(result$scores), unlist(result$labels))
    
    # Map full label text back to short names
    active <- character(0)
    for (i in seq_along(labels)) {
      if (scores[labels[i]] >= threshold) {
        active <- c(active, short_names[i])
      }
    }
    
    # Build combined partisan type label (sorted for consistency)
    if (length(active) == 0) {
      partisan_type <- "unknown"
    } else {
      active_sorted <- sort(active)  # alphabetical: centre, left, right
      partisan_type <- paste(active_sorted, collapse = "_and_")
    }
    
    # Best single score for confidence reporting
    best_score <- max(unlist(result$scores))
    
    data.frame(
      predicted_partisan_type = partisan_type,
      confidence = best_score,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(predicted_partisan_type = "unknown", confidence = 0,
               stringsAsFactors = FALSE)
  })
}

# Subset to unknown events with usable notes
unknown_events <- italy_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n")

# Apply classifier — will take a while on CPU
text_classifications <- unknown_events %>%
  mutate(
    classification_result = map(
      notes,
      ~classify_text_multi(.x, classifier, candidate_labels, score_threshold, label_short)
    ),
    predicted_partisan_type = map_chr(classification_result, "predicted_partisan_type"),
    confidence              = map_dbl(classification_result, "confidence"),
    propagation_source = if_else(
      predicted_partisan_type != "unknown",
      "text_classification",
      "unknown"
    )
  ) %>%
  select(event_id_cnty, predicted_partisan_type, confidence, propagation_source)

################################################################################
# STEP 3: MERGE AND PRODUCE FINAL CLASSIFICATION
################################################################################

italy_acled_df <- italy_acled_df %>%
  left_join(
    text_classifications %>%
      select(event_id_cnty, predicted_partisan_type, confidence, propagation_source),
    by = "event_id_cnty"
  ) %>%
  mutate(
    event_partisan_type_final = case_when(
      event_partisan_type != "unknown"                                          ~ event_partisan_type,
      !is.na(predicted_partisan_type) & predicted_partisan_type != "unknown"   ~ predicted_partisan_type,
      TRUE                                                                      ~ "unknown"
    ),
    classification_source = case_when(
      event_partisan_type != "unknown"         ~ "original",
      propagation_source == "text_classification" ~ "text_classification",
      TRUE                                     ~ "unknown"
    )
  ) %>%
  select(-propagation_source)

cat("\nFinal classification breakdown:\n")
italy_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(italy_acled_df$event_partisan_type_final == "unknown"), "\n")

################################################################################
# SAVE
################################################################################

write_csv(italy_acled_df, "~/Downloads/italy_acled_partisan_classification_bootstrapped.csv")


################################################################################
# SAVE
################################################################################

write_csv(italy_acled_df, "~/Downloads/italy_acled_partisan_classification_bootstrapped.csv")

#################################################################################
# Validation 
#################################################################################

text_classified_obs <- italy_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/italy_text_class_random_sample.xlsx")

################################################################################

candidate_labels <- c(
  "a left-wing protest about labor unions, workers rights, housing,
   student occupations, precarious work, public services, migrant
   workers, anti-racism, or anti-fascism",
  
  "a right-wing protest about the green pass, vaccine mandates,
   coronavirus restrictions, immigration, national sovereignty,
   law and order, or EU farming regulations",
  
  "a local community protest about a specific planning or
   infrastructure decision with no broader political ideology"
)

# Short names used in label_map and output columns
label_short <- c("left", "right", "centre")

# Threshold for a label to be counted as present (multi-label)
score_threshold <- 0.45

classify_text_multi <- function(text, classifier, labels, threshold, short_names) {
  tryCatch({
    result <- classifier(text, labels, multi_label = TRUE)
    
    scores <- setNames(unlist(result$scores), unlist(result$labels))
    
    active <- character(0)
    for (i in seq_along(labels)) {
      if (scores[labels[i]] >= threshold) {
        active <- c(active, short_names[i])
      }
    }
    
    if (length(active) == 0) {
      partisan_type <- "unknown"
    } else {
      active_sorted <- sort(active)  # alphabetical: centre, left, right
      partisan_type <- paste(active_sorted, collapse = "_and_")
    }
    
    best_score <- max(unlist(result$scores))
    
    data.frame(
      predicted_partisan_type = partisan_type,
      confidence = best_score,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(predicted_partisan_type = "unknown", confidence = 0,
               stringsAsFactors = FALSE)
  })
}

unknown_events <- italy_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 10713 to class 

text_classifications <- unknown_events %>%
  mutate(
    classification_result = map(
      notes,
      ~classify_text_multi(.x, classifier, candidate_labels, score_threshold, label_short)
    ),
    predicted_partisan_type = map_chr(classification_result, "predicted_partisan_type"),
    confidence              = map_dbl(classification_result, "confidence"),
    propagation_source = if_else(
      predicted_partisan_type != "unknown",
      "text_classification",
      "unknown"
    )
  ) %>%
  select(event_id_cnty, predicted_partisan_type, confidence, propagation_source)

italy_acled_df <- italy_acled_df %>%
  left_join(
    text_classifications %>% select(event_id_cnty, predicted_partisan_type, confidence, propagation_source),
    by = "event_id_cnty"
  ) %>%
  mutate(
    event_partisan_type_final = case_when(
      event_partisan_type != "unknown" ~ event_partisan_type,
      !is.na(predicted_partisan_type) & predicted_partisan_type != "unknown" ~ predicted_partisan_type,
      TRUE ~ "unknown"
    ),
    classification_source = case_when(
      event_partisan_type != "unknown" ~ "original",
      propagation_source == "text_classification" ~ "text_classification",
      TRUE ~ "unknown"
    )
  ) %>%
  select(-propagation_source)

cat("\nFinal classification breakdown:\n")
italy_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(italy_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(italy_acled_df, "~/Downloads/italy_acled_partisan_classification_bootstrapped3.csv")

text_classified_obs <- italy_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/italy_text_class_random_sample3.xlsx")

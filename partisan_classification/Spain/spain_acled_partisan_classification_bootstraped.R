# Title: Partisanship Classification - Addendum (NLP)
# Author: Katelyn Nutley
# Date: 24-02-2026
# Description: NLP-based bootstrap to classify unknown actors in SPAIN ACLED data.
#              For mixed events (classified + unknown assoc_actors co-occurring),
#              uses the notes field alongside the classified actor as a partisan 
#              anchor to contextualise unknown actors via zero-shot classification.
#              Requires: spain_acled_partisan_classification.csv from main script.

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
spain_acled_df <- read_csv("~/Downloads/spain_acled_partisan_classification.csv")

################################################################################
# STEP 1: IDENTIFY MIXED EVENTS
################################################################################

# The logic here is: if an event has at least one classified actor and at least
# one unknown actor, then we can use NLP techniques on the text field to contextualise 
# political behaviour.

spain_acled_df <- spain_acled_df %>%
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

cat("Mixed events for NLP classification:", sum(spain_acled_df$mixed == 1), "\n")

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

candidate_labels <- c(
  "left-wing or progressive causes such as anti-racism, environmentalism, workers rights, or social justice",
  "right-wing or conservative causes such as nationalism, anti-immigration, or traditional values",
  "centrist or non-partisan causes such as electoral reform or local community issues"
)

label_map <- c(
  "left-wing or progressive causes such as anti-racism, environmentalism, workers rights, or social justice" = "left_only",
  "right-wing or conservative causes such as nationalism, anti-immigration, or traditional values" = "right_only",
  "centrist or non-partisan causes such as electoral reform or local community issues" = "centre_only"
)

classify_text <- function(text, classifier, labels) {
  tryCatch({
    result <- classifier(text, labels, multi_label = FALSE)
    data.frame(
      predicted_label = result$labels[[1]],
      confidence = result$scores[[1]]
    )
  }, error = function(e) {
    data.frame(predicted_label = "unknown", confidence = 0)
  })
}

# Subset to unknown events with usable notes
unknown_events <- spain_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n")

# Apply classifier - will take a while on CPU
text_classifications <- unknown_events %>%
  mutate(
    classification_result = map(notes, ~classify_text(.x, classifier, candidate_labels)),
    predicted_label_raw = map_chr(classification_result, "predicted_label"),
    confidence = map_dbl(classification_result, "confidence"),
    predicted_partisan_type = recode(predicted_label_raw, !!!label_map)
  ) %>%
  select(event_id_cnty, predicted_partisan_type, confidence)

# Confidence threshold - adjust based on inspection
confidence_threshold <- 0.45

text_classifications <- text_classifications %>%
  mutate(
    predicted_partisan_type = if_else(
      confidence >= confidence_threshold,
      predicted_partisan_type,
      "unknown"
    ),
    propagation_source = if_else(
      predicted_partisan_type != "unknown",
      "text_classification",
      "unknown"
    )
  )

################################################################################
# STEP 3: MERGE AND PRODUCE FINAL CLASSIFICATION
################################################################################

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
  "a left-wing protest about workers rights, trade union organising, anti-racism, feminism, housing rights, Palestine solidarity, regional autonomy, environmental activism, antifascist action, or anti-austerity",
  "a right-wing protest about immigration, Vox politics, Spanish nationalism, anti-feminism, coronavirus restrictions, vaccine mandates, traditional Catholic values, anti-Islam positions, or far-right and identitarian movements",
  "a local community protest about a specific planning or infrastructure decision with no broader political ideology"
)

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

unknown_events <- spain_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 11380 to class 

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

spain_acled_df <- spain_acled_df %>%
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
spain_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(spain_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(spain_acled_df, "~/Downloads/spain_acled_partisan_classification_bootstrapped3.csv")

text_classified_obs <- spain_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/spain_text_class_random_sample3.xlsx")

################################################################################

spain_acled_df <- spain_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Counter-Vox / counter-far-right protests misclassified as right → left
      str_detect(notes, regex("vox|catalan alliance|aliança catalana|orriols|stop mare mortum|fascis", 
                              ignore_case = TRUE)) &
        str_detect(notes, regex("against|protest|oppos|counter|anti|condemn|jeer", 
                                ignore_case = TRUE)) &
        event_partisan_type_final == "right" ~ "left",
      
      # Police / occupational sectoral disputes misclassified as left → centre
      str_detect(notes, regex("police officer|ertzaintza|transport worker|camionero|trucker|market worker|vendedor ambulante", 
                              ignore_case = TRUE)) &
        str_detect(notes, regex("salary|wage|working conditions|pay|labor agreement|convenio", 
                                ignore_case = TRUE)) &
        event_partisan_type_final == "left" ~ "centre",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex("vox|catalan alliance|aliança catalana|orriols|stop mare mortum|fascis", ignore_case = TRUE)) &
        str_detect(notes, regex("against|protest|oppos|counter|anti|condemn|jeer", ignore_case = TRUE)) &
        classification_source != "original" ~ "rule_correction",
      str_detect(notes, regex("police officer|ertzaintza|transport worker|camionero|trucker|market worker|vendedor ambulante", ignore_case = TRUE)) &
        str_detect(notes, regex("salary|wage|working conditions|pay|labor agreement|convenio", ignore_case = TRUE)) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

write_csv(spain_acled_df, "~/Downloads/spain_acled_partisan_classification_bootstrapped4.csv")

text_classified_obs <- spain_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/spain_text_class_random_sample4.xlsx")

################################################################################
# marginal improvement but still annoyed with the errors: 

spain_acled_df <- read_csv("~/Downloads/spain_acled_partisan_classification_bootstrapped3.csv")

spain_acled_df <- spain_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Counter-far-right protests misclassified as right → left
      str_detect(notes, regex(
        "vox|catalan alliance|aliança catalana|orriols|stop mare mortum|fascis|immigration forum|expulsion order|melilla",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" ~ "left",
      
      # Occupational/sectoral disputes misclassified as left → centre
      str_detect(notes, regex(
        "police officer|ertzaintza|transport worker|camionero|trucker|market worker|vendedor ambulante|hospitality|hostelería",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "salary|wage|working conditions|pay|labor agreement|convenio|economic situation",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left" ~ "centre",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex(
        "vox|catalan alliance|aliança catalana|orriols|stop mare mortum|fascis|immigration forum|expulsion order|melilla",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      str_detect(notes, regex(
        "police officer|ertzaintza|transport worker|camionero|trucker|market worker|vendedor ambulante|hospitality|hostelería",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "salary|wage|working conditions|pay|labor agreement|convenio|economic situation",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

write_csv(spain_acled_df, "~/Downloads/spain_acled_partisan_classification_bootstrapped5.csv")

text_classified_obs <- spain_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/spain_text_class_random_sample5.xlsx")

#################################################################################



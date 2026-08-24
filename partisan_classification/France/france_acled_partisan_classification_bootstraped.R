# Title: France Partisanship Classification - Addendum (NLP Bootstrap)
# Author: Katelyn Nutley
# Date: 23-02-2026
# Description: NLP-based bootstrap to classify unknown actors in FRANCE ACLED data.
#              For mixed events (classified + unknown assoc_actors co-occurring),
#              uses the notes field alongside the classified actor as a partisan 
#              anchor to contextualise unknown actors via zero-shot classification.
#              Requires: france_acled_partisan_classification.csv from main script.

library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(reticulate)

reticulate::py_install("transformers")
reticulate::py_install("torch")

# Load output from main classification script
france_acled_df <- read_csv("~/Downloads/france_acled_partisan_classification.csv")

################################################################################
# STEP 1: IDENTIFY MIXED EVENTS
################################################################################

# The logic here is: if an event has at least one classified actor and at least
# one unknown actor, then we can use NLP techniques on the text field to contextualise 
# political behaviour.

france_acled_df <- france_acled_df %>%
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

################################################################################
# STEP 2: ZERO-SHOT TEXT CLASSIFICATION
################################################################################

# Point to your Python environment
# use_condaenv("your_env_name")
# use_virtualenv("path/to/venv")

transformers <- import("transformers")
torch <- import("torch")

classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli"
)

# Labels written for French protest context:
#
# LEFT:   French left protests code via trade union organising (CGT, CFDT,
#         FO), anti-racism (SOS Racisme, LICRA), Palestine solidarity,
#         pension and public service protection, housing rights,
#         environmental activism (ANV-COP21, Youth for Climate),
#         anti-austerity, and feminist movements.
#
# RIGHT:  French right protests code via anti-immigration, nationalism
#         and identitarian movements (RN-adjacent, Génération Identitaire),
#         anti-coronavirus health pass or vaccine mandates, anti-Islam,
#         pro-police and law and order, or anti-green regulation.
#
# CENTRE: French centrist protests code via professional or business sector
#         economic grievances (farmers, small traders, liberal professions),
#         civic or non-partisan single-issue campaigns, or local community
#         issues outside the left-right spectrum. Note: gilets jaunes events
#         should be coded as left_and_right given their cross-partisan nature.

candidate_labels <- c(
  "left-wing protest for causes such as workers rights, trade union
   organising, anti-racism, Palestine solidarity, pension and public
   service protection, housing rights, environmental activism,
   feminist movements, or anti-austerity",
  "right-wing protest against causes such as immigration or asylum
   seekers, in favour of nationalism or identitarian movements,
   against coronavirus health pass or vaccine mandates, against
   Islam, in favour of law and order or pro-police positions,
   or against green regulations",
  "centrist or non-partisan protest for causes such as professional
   or business sector economic relief, farmers grievances, liberal
   profession demands, civic safety, or single-issue campaigns
   outside the left-right spectrum"
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

# Subset to unknown events with usable notes
unknown_events <- france_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # wow, 16,173 here 

# Apply classifier - will take a while on CPU, esp. with how many are in the french case 
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

france_acled_df <- france_acled_df %>%
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
france_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

################################################################################
# SAVE
################################################################################

write_csv(france_acled_df, "~/Downloads/france_acled_partisan_classification_bootstrapped1.csv")

#################################################################################
# Validation 
#################################################################################

text_classified_obs <- france_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/france_text_class_random_sample1.xlsx") # indicative of the fact
# that I'm trying to iteratively refine this 

#################################################################################
# Revision
#################################################################################

# Right, so that one got 20% right. So the labels need to be refined. Here we go: 

# reload dataset and rerun: 

candidate_labels <- c(
  "left-wing protest for causes such as pension reform opposition,
   trade union strike action, workers demanding pay rises or better
   conditions, healthcare workers including nurses or midwives
   demanding recognition or pay, public service protection, teachers
   opposing education reform, LGBTQI+ rights or pride marches,
   feminist or women's rights protests, anti-racism, Palestine
   solidarity, diaspora pro-democracy protests, environmental or
   climate activism, or anti-austerity",
  "right-wing protest against coronavirus health pass or vaccination
   pass requirements, against compulsory vaccination, in favour of
   law and order or pro-police positions, by hunters defending
   hunting rights, pro-military or security forces, religious
   conservative causes such as anti-abortion or anti-euthanasia,
   against immigration or demanding evacuation of migrant camps,
   in favour of nationalism or identitarian movements, or farmers
   protesting against EU environmental or agricultural regulations",
  "centrist or non-partisan protest for causes such as local
   planning decisions, opposition to specific local developments,
   professional or business sector economic relief, small trader
   or liberal profession demands, civic safety, or single-issue
   community campaigns with no clear left-right alignment"
)

label_short <- c("left", "right", "centre") 

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

unknown_events <- france_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # wow, 16,173 here 

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

france_acled_df <- france_acled_df %>%
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
france_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(france_acled_df, "~/Downloads/france_acled_partisan_classification_bootstrapped2.csv")

text_classified_obs <- france_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/france_text_class_random_sample2.xlsx") 
# Honestly, I didn't even check these; just like had to refine it based on what I know 
# from the UK stuff 

###############################################################################

candidate_labels <- c(
  "a left-wing protest about pensions, trade unions, pay, public
   services, healthcare workers, teachers, LGBTQI+ rights, feminism,
   anti-racism, Palestine, the environment, or anti-austerity",
  
  "a right-wing protest about immigration, the passe sanitaire,
   vaccine mandates, nationalism, anti-Islam, hunting rights,
   law and order, or EU farming regulations",
  
  "a local community protest about a specific planning or
   infrastructure decision with no broader political ideology"
)

label_short <- c("left", "right", "centre") 

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

unknown_events <- france_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # wow, 16,173 here 

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

france_acled_df <- france_acled_df %>%
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
france_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(france_acled_df, "~/Downloads/france_acled_partisan_classification_bootstrapped3.csv")

text_classified_obs <- france_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/france_text_class_random_sample3.xlsx") 
# Checking nowww! 

#################################################################################

france_acled_df <- read_csv("~/Downloads/france_acled_partisan_classification_bootstrapped3.csv")

france_acled_df <- france_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Pro-immigration / anti-far-right protests misclassified as right → left
      str_detect(notes, regex(
        "xenophob|racis|anti-immigration bill|passe sanitaire|loi immigration",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|reject|denounc|call on|deemed",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" ~ "left",
      
      # Liberal profession / sectoral demands misclassified as left → centre
      str_detect(notes, regex(
        "lawyer|avocat|barrister|ambulance|court clerk|greffier|notaire|notary",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "salary|wage|pay|conditions|status|resources|regime|pension",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left" ~ "centre",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex(
        "xenophob|racis|anti-immigration bill|passe sanitaire|loi immigration",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|reject|denounc|call on|deemed",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      str_detect(notes, regex(
        "lawyer|avocat|barrister|ambulance|court clerk|greffier|notaire|notary",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "salary|wage|pay|conditions|status|resources|regime|pension",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

write_csv(france_acled_df, "~/Downloads/france_acled_partisan_classification_bootstrapped4.csv")

text_classified_obs <- france_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>%
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/france_text_class_random_sample4.xlsx")

################################################################################
# fixing the discrepancies: 

library(readr)
library(dplyr)
library(stringr)
library(writexl)

france_acled_df <- read_csv("~/Downloads/france_acled_partisan_classification_bootstrapped3.csv")

france_acled_df <- france_acled_df %>%
  mutate(
    matched_rule_left = (
      str_detect(notes, regex(
        "xenophob|racis|anti-immigration bill|passe sanitaire|loi immigration",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|reject|denounc|call on|deemed",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" &
        classification_source != "original"
    ),
    matched_rule_centre = (
      str_detect(notes, regex(
        "lawyer|avocat|barrister|ambulance|court clerk|greffier|notaire|notary",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "salary|wage|pay|conditions|status|resources|regime|pension",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left" &
        classification_source != "original"
    ),
    event_partisan_type_final = case_when(
      matched_rule_left   ~ "left",
      matched_rule_centre ~ "centre",
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      matched_rule_left   ~ "rule_correction",
      matched_rule_centre ~ "rule_correction",
      TRUE ~ classification_source
    )
  ) %>%
  select(-matched_rule_left, -matched_rule_centre)

# Check discrepancy is gone
discrepancy <- france_acled_df %>%
  filter(classification_source != "unknown" & event_partisan_type_final == "unknown")
nrow(discrepancy)

table(france_acled_df$classification_source)
table(france_acled_df$event_partisan_type_final)

write_csv(france_acled_df, "~/Downloads/france_acled_partisan_classification_bootstrapped5.csv")

random_sample <- france_acled_df %>%
  filter(classification_source == "text_classification") %>%
  slice_sample(n = 100)
write_xlsx(random_sample, "~/Documents/france_text_class_random_sample5.xlsx")


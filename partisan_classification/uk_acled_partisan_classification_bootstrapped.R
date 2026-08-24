# Title: UK Partisanship Classification - Addendum (NLP Bootstrap)
# Author: Katelyn Nutley
# Date: 23-02-2026
# Description: NLP-based bootstrap to classify unknown actors in UK ACLED data.
#              For mixed events (classified + unknown assoc_actors co-occurring),
#              uses the notes field alongside the classified actor as a partisan 
#              anchor to contextualise unknown actors via zero-shot classification.
#              Requires: uk_acled_partisan_classification.csv from main script.

library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(reticulate)

reticulate::py_install("transformers")
reticulate::py_install("torch")

# Load output from main classification script
uk_acled_df <- read_csv("~/Downloads/uk_acled_partisan_classification.csv")

################################################################################
# STEP 1: IDENTIFY MIXED EVENTS
################################################################################

# The logic here is: if an event has at least one classified actor and at least
# one unknown actor, then we can use NLP techniques on the text field to contextualise 
# political behaviour.

uk_acled_df <- uk_acled_df %>%
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

# Point to your Python environment - update path as needed for diff countries, Katie 
# use_condaenv("your_env_name")
# use_virtualenv("path/to/venv")

transformers <- import("transformers")
torch <- import("torch")

classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli"
) # this was chosen because it's free and also as it's trained on real blurbs, probably fairly accurate 

# Labels written for UK protest context:
#
# LEFT:   UK left protests code via trade union organising, anti-racism,
#         Palestine solidarity, NHS and public service protection, housing
#         rights, environmental activism (JSO, XR), anti-austerity.
#
# RIGHT:  UK right protests code via anti-immigration/asylum, nationalism,
#         pro-Brexit, anti-coronavirus restrictions/vaccine mandates,
#         anti-ULEZ/green regulations, far-right and anti-Islam positions.
#
# CENTRE: UK centrist protests code via business sector economic relief,
#         civic safety, local community issues, single-issue campaigns
#         outside the left-right spectrum.

candidate_labels <- c(
  "left-wing protest for causes such as workers rights, trade union
   organising, anti-racism, Palestine solidarity, NHS and public
   service protection, housing and tenants rights, environmental
   activism such as Just Stop Oil or Extinction Rebellion,
   or anti-austerity",
  "right-wing protest against causes such as immigration or asylum
   seekers, in favour of nationalism or Brexit, against coronavirus
   restrictions or vaccine mandates, against ULEZ or green
   regulations, or promoting far-right or anti-Islam positions",
  "centrist or non-partisan protest for causes such as business
   sector economic relief, civic safety, road safety, local
   community issues, or single-issue campaigns outside the
   left-right spectrum"
)

label_short <- c("left", "right", "centre")

# Threshold for a label to be counted as present (multi-label)
score_threshold <- 0.45 # I messed around with this and thought that 0.45 was the least 
# restrictive without letting through too much noise 


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
unknown_events <- uk_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 2,868 isn't bad! 

# Apply classifier - will take a while on CPU; maybe go work on something else and come back 
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

uk_acled_df <- uk_acled_df %>%
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
uk_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

################################################################################
# SAVE
################################################################################

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped1.csv")

#################################################################################
# Validation 
#################################################################################

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample1.xlsx") #added 1 bc 
# this is v2, the first version was at 85%, but I took a look at the type 1 errors and 
# they were clustered around some specific protest types, so I fixed this 

# Okay, so I did the second version and it was at like 67%; so I clustered the errors: 
# 1) farmers (11 errors, biggest problem) - labeled as centre or centre_and_left 
# when it should be centre_and_right or right 
# 2) anti-lockdown - should be right
# 3) foreign affairs (Burma, Darfur, Iran, rent strike) - should all be cleanly left, 
# but, they're coded as centre or centre_and_left 
# 4) NIMBY-ism - overcoded as left; 

#################################################################################
# Revision 
#################################################################################

# Revised labels with reload + run: 

candidate_labels <- c(
  "left-wing protest for causes such as workers rights, trade union
   organising, pay demands, rent strikes, anti-racism, diaspora
   solidarity or human rights protests, Palestine solidarity,
   NHS and public service protection, feminist or women's rights
   activism, environmental activism such as Just Stop Oil or
   Extinction Rebellion, or anti-austerity",
  "right-wing protest against causes such as immigration or asylum
   seekers, in favour of nationalism or Brexit, against coronavirus
   restrictions or vaccine mandates, against ULEZ or green
   regulations, farmers protesting against government agricultural
   or inheritance tax policy, or promoting far-right or
   anti-Islam positions",
  "centrist or non-partisan protest for causes such as local
   planning decisions, opposition to specific developments,
   business sector economic relief, civic safety, road safety,
   or single-issue community campaigns with no clear
   left-right alignment"
)

label_short <- c("left", "right", "centre")

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

unknown_events <- uk_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

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

uk_acled_df <- uk_acled_df %>%
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
uk_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped2.csv") # this is the refined csv 

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample2.xlsx") # let's see if it worked; 

# Okay, so there was 48% accuracy; clusters: 
# 1) left wing causes coded as centre (25 errors) - The model doesn't treat LGBT+, animal rights, 
# diaspora solidarity, trans rights, facial recognition / civil liberties, and teachers' pay as 
# inherently left-wing. And I think that was the issue for v2's collapse. 
# 2) Centre bleeding into centre_and_left (19 errors) - Even when the model gets the left signal, 
# it still scores centre above 0.45 on clearly left events (climate activists, NHS workers, rent
# strikes, cycling protests, anti-racism marches).
# 3) Anti-lockdown coded as centre or left? 

# The v2 revision narrowed the centre label (good) but it didn't expand the left label to cover the
# specific causes the model was missing. So accuracy dropped by 19 points... 


#################################################################################
# Revision 2 
#################################################################################

# revised labels: 

candidate_labels <- c(
    "a left-wing protest about workers rights, pay, housing, racism, 
   LGBT+ rights, trans rights, feminism, animal rights, the environment, 
   civil liberties, asylum seekers, or diaspora solidarity",
    
    "a right-wing protest about immigration, nationalism, Brexit, 
   coronavirus restrictions, vaccine mandates, farming policy, 
   ULEZ, net zero, far-right causes, or anti-Islam positions",
    
    "a local community protest about a specific planning or 
   infrastructure decision with no broader political ideology"
  )

label_short <- c("left", "right", "centre")

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

unknown_events <- uk_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

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

uk_acled_df <- uk_acled_df %>%
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
uk_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(uk_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped3.csv") # this is the refined csv 

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample3.xlsx") # let's see if it worked; 83% correct
# now to do spot checks... 

#################################################################################

uk_acled_df <- read_csv("~/Downloads/uk_acled_partisan_classification_bootstrapped3.csv")

uk_acled_df <- uk_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Counter-far-right / anti-Brexit protests misclassified as right → left
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|remain|hostile environment|citizenship amendment|CAA",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" ~ "left",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|remain|hostile environment|citizenship amendment|CAA",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped4.csv")

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>%
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample4.xlsx")

##################################################################################

uk_acled_df <- read_csv("~/Downloads/uk_acled_partisan_classification_bootstrapped3.csv")

uk_acled_df <- uk_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Counter-far-right / anti-Brexit protests misclassified as right → left
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|no.deal|hostile environment|citizenship amendment|CAA|borders.*bill|nationality.*bill|scottish independence",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" ~ "left",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|no.deal|hostile environment|citizenship amendment|CAA|borders.*bill|nationality.*bill|scottish independence",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl",
          ignore_case = TRUE
        )) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped5.csv")

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>%
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample5.xlsx")

###############################################################################
# going to just correct this one more time bc UK has so few obs 

uk_acled_df <- read_csv("~/Downloads/uk_acled_partisan_classification_bootstrapped3.csv")

uk_acled_df <- uk_acled_df %>%
  mutate(
    matched_rule = (
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|brexit|no.deal|hostile environment|citizenship amendment|CAA|borders.*bill|nationality.*bill|scottish independence",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl|campaign",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" &
        classification_source != "original"
    ),
    event_partisan_type_final = case_when(
      matched_rule ~ "left",
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      matched_rule ~ "rule_correction",
      TRUE ~ classification_source
    )
  ) %>%
  select(-matched_rule)

write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped6.csv")

text_classified_obs <- uk_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>%
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample6.xlsx")

################################################################################

library(readr)
library(dplyr)
library(stringr)
library(writexl)

uk_acled_df <- read_csv("~/Downloads/uk_acled_partisan_classification_bootstrapped3.csv")

uk_acled_df <- uk_acled_df %>%
  mutate(
    matched_rule = (
      str_detect(notes, regex(
        "reform uk|nigel farage|edl|britain first|patriotic alternative|far-right|anti-brexit|brexit|no.deal|hostile environment|citizenship amendment|CAA|borders.*bill|nationality.*bill|scottish independence",
        ignore_case = TRUE
      )) &
        str_detect(notes, regex(
          "against|protest|oppos|counter|anti|condemn|jeer|complain|reject|denounc|vigil|heckl|campaign",
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "right" &
        classification_source != "original"
    ),
    event_partisan_type_final = case_when(
      matched_rule ~ "left",
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      matched_rule ~ "rule_correction",
      TRUE ~ classification_source
    )
  ) %>%
  select(-matched_rule)

# Check
table(uk_acled_df$classification_source)
table(uk_acled_df$event_partisan_type_final)

# Correction
uk_acled_df <- uk_acled_df %>%
  mutate(event_partisan_type_final = recode(event_partisan_type_final,
                                            "left"   = "left_only",
                                            "right"  = "right_only",
                                            "centre" = "centre_only",
                                            "left_vs_centre" = "left_and_centre",
                                            "left_vs_right" = "left_and_right",
                                            "right_vs_centre" = "centre_and_right"
  ))

table(uk_acled_df$event_partisan_type_final)

# Save
write_csv(uk_acled_df, "~/Downloads/uk_acled_partisan_classification_bootstrapped7.csv")

# Redraw validation sample
random_sample <- uk_acled_df %>%
  filter(classification_source == "text_classification") %>%
  slice_sample(n = 100)

write_xlsx(random_sample, "~/Documents/uk_text_class_random_sample7.xlsx")


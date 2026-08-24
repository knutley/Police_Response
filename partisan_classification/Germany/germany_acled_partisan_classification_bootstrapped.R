# Title: Germany Partisanship Classification - Addendum (NLP Bootstrap)
# Author: Katelyn Nutley
# Date: 23-02-2026
# Description: NLP-based bootstrap to classify unknown actors in GERMANY ACLED data.
#              For mixed events (classified + unknown assoc_actors co-occurring),
#              uses the notes field alongside the classified actor as a partisan 
#              anchor to contextualise unknown actors via zero-shot classification.
#              Requires: germany_acled_partisan_classification.csv from main script.

library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(reticulate)

reticulate::py_install("transformers")
reticulate::py_install("torch")

# Load output from main classification script
germany_acled_df <- read_csv("~/Downloads/germany_acled_partisan_classification.csv")

################################################################################
# STEP 1: IDENTIFY MIXED EVENTS
################################################################################

# The logic here is: if an event has at least one classified actor and at least
# one unknown actor, then we can use NLP techniques on the text field to contextualise 
# political behaviour.

germany_acled_df <- germany_acled_df %>%
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

# Point to your Python environment - update path as needed
# use_condaenv("your_env_name")
# use_virtualenv("path/to/venv")

transformers <- import("transformers")
torch <- import("torch")

classifier <- transformers$pipeline(
  "zero-shot-classification",
  model = "facebook/bart-large-mnli"
)

# Labels written for German protest context:
#
# LEFT:   German left protests code via DGB-affiliated union organising,
#         anti-racism, anti-AfD and pro-democracy movements, environmental
#         activism (Fridays for Future, Ende Gelände, Last Generation),
#         Palestine solidarity, anti-austerity, and antifascist action.
#
# RIGHT:  German right protests code via anti-immigration, AfD-adjacent
#         nationalism, PEGIDA and anti-Islam movements, Querdenken and
#         anti-coronavirus restrictions, pro-Russia/anti-NATO positions,
#         and far-right or identitarian movements.
#
# CENTRE: German centrist protests code via farmer and agricultural
#         grievances (agrardiesel, tractor protests), business sector
#         economic relief, motorbike and vehicle regulation protests,
#         and civic or single-issue campaigns outside the left-right
#         spectrum.

candidate_labels <- c(
  "left-wing protest for causes such as workers rights, trade union
   organising, anti-racism, anti-AfD or pro-democracy movements,
   Palestine solidarity, environmental activism such as Fridays for
   Future or Last Generation, antifascist action, or anti-austerity",
  "right-wing protest against causes such as immigration or asylum
   seekers, in favour of nationalism or AfD politics, against
   coronavirus restrictions or vaccine mandates, promoting PEGIDA
   or anti-Islam positions, pro-Russia or anti-NATO, or far-right
   and identitarian movements",
  "centrist or non-partisan protest for causes such as farmer or
   agricultural grievances, business sector economic relief,
   motorbike or vehicle regulation protests, civic safety, or
   single-issue campaigns outside the left-right spectrum"
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
unknown_events <- germany_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 14,229 total 

# Apply classifier - will take a while on CPU
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

germany_acled_df <- germany_acled_df %>%
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
germany_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(germany_acled_df$event_partisan_type_final == "unknown"), "\n")

################################################################################
# SAVE
################################################################################

write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped1.csv")

#################################################################################
# Validation 
#################################################################################

text_classified_obs <- germany_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample1.xlsx")

# So, after doing the validation check, Germany was roughly 50%. This is much lower than
# the UK and France's 85%. So, I 'm going to make some post-BART corrections. The majority
# of the issues arose from corona-virus regulations demonstrations

germany_acled_df <- germany_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      classification_source == "text_classification" &
        str_detect(str_to_lower(notes), 
                   "coronavirus-related measures|querdenken|contact ban|
         lockdown measures|health pass|impfpflicht|corona-massnahmen") &
        event_partisan_type_final == "left_only" ~ "right_only",
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      # Flag corrected observations separately for transparency
      classification_source == "text_classification" &
        str_detect(str_to_lower(notes),
                   "coronavirus-related measures|querdenken|contact ban|
         lockdown measures|health pass|impfpflicht|corona-massnahmen") &
        event_partisan_type_final == "right_only" ~ "text_classification_corrected",
      TRUE ~ classification_source
    )
  )

text_classified_obs2 <- germany_acled_df %>%
  filter(classification_source == c("text_classification_corrected", "text_classification"))
random_sample2 <- text_classified_obs2 %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample2, "~/Documents/germany_text_class_random_sample2.xlsx")

# Having corrected for some of this, there was still an issue with the corona pattern, 
# the farmer patterns and the antiafd pattern (71%); going to correct once more: 

germany_acled_df <- germany_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Anti-coronavirus measures protests miscoded as left_only -> correct to right_only
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(str_to_lower(notes), 
                   "coronavirus protection measures|coronavirus lockdown|corona-massnahmen|
           impfpflicht|covid-massnahmen|lockdown measures|contact ban|health pass|
           querdenken|coronavirus-related measures|compulsory vacc|
           corona schutzmasnahmen|against the coronavirus|covid protection|
           corona measures|impfung|walk.*corona|corona.*walk") &
        event_partisan_type_final == "left_only" ~ "right_only",
      
      # Farmer protests miscoded as left_only -> correct to centre_only
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(str_to_lower(notes),
                   "agrardiesel|agricultural subsid|agrarian polic|farmers.*protest|
           bauern|agricultural polic|agrarian package|farmers.*tractor|
           tractor.*protest|farmers.*blockade|farmers.*demonstration|
           vehicle tax exemption|farmers and craftsmen|farmers.*nationwide|
           nationwide farmers") &
        event_partisan_type_final == "left_only" ~ "centre_only",
      
      # Anti-AfD and pro-democracy protests miscoded as right_only -> correct to left_only
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(str_to_lower(notes),
                   "against.*afd|gegen.*afd|far-right shift|against far-right|
           lights against the right|defend.*democracy|against the rise of far-right|
           against far-right ideolog|against right-wing|sea of lights|
           bundnis fur demokratie|alliance for democracy|protest.*afd|
           afd.*protest|against.*afd.*politics|wider protest movement.*far-right|
           nationwide protest movement.*defend|right-wing.*activist|
           against.*sellner|against.*appearance.*right") &
        event_partisan_type_final == "right_only" ~ "left_only",
      
      # Motorbike/vehicle protests miscoded as left_only -> correct to centre_only  
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(str_to_lower(notes),
                   "motorbike.*protest|motorcycle.*protest|biker.*protest|ffmc|
           bikers.*gathered|motorbike.*ban|weekend.*driving ban|
           technical control.*motorbike|motorbike.*technical|
           ride free|motor biker") &
        event_partisan_type_final == "left_only" ~ "centre_only",
      
      TRUE ~ event_partisan_type_final
    ),
    
    # Update classification source to flag corrections
    classification_source = case_when(
      classification_source == "text_classification" &
        event_partisan_type_final != lag(event_partisan_type_final) ~ "text_classification_corrected",
      TRUE ~ classification_source
    )
  )
text_classified_obs3 <- germany_acled_df %>%
  filter(classification_source == c("text_classification_corrected", "text_classification"))
random_sample3 <- text_classified_obs3 %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample3, "~/Documents/germany_text_class_random_sample3.xlsx")

##################################################################################

# Labels written for German protest context:
#
# LEFT:   German left protests code via workers rights, trade union
#         organising, anti-racism, anti-AfD or pro-democracy movements,
#         Palestine solidarity, environmental activism, antifascist
#         action, and anti-austerity.
#
# RIGHT:  German right protests code via anti-immigration, AfD-adjacent
#         nationalism, PEGIDA, anti-Islam, Querdenken and anti-coronavirus
#         restrictions, pro-Russia or anti-NATO, and far-right or
#         identitarian movements.
#
# CENTRE: German centrist protests code via local planning or
#         infrastructure decisions with no broader political ideology.

candidate_labels <- c(
  "a left-wing protest about workers rights, trade union organising,
   anti-racism, anti-AfD or pro-democracy movements, Palestine solidarity,
   environmental activism, antifascist action, or anti-austerity",
  "a right-wing protest about immigration, AfD politics, nationalism,
   PEGIDA, anti-Islam positions, coronavirus restrictions, vaccine mandates,
   pro-Russia or anti-NATO causes, or far-right and identitarian movements",
  "a local community protest about a specific planning or
   infrastructure decision with no broader political ideology"
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

unknown_events <- germany_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 14229 to class 

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

germany_acled_df <- germany_acled_df %>%
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
germany_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(italy_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped3.csv")

text_classified_obs <- germany_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample3.xlsx")

##################################################################################

candidate_labels <- c(
  "a left-wing protest defending democracy, opposing the AfD or NPD, opposing PEGIDA, opposing far-right extremism or remigration, or advocating for workers rights, anti-racism, Palestine solidarity, environmental activism, antifascist action, pro-refugee or anti-deportation causes, or anti-austerity",
  "a right-wing protest organised by or in support of the AfD, NPD, PEGIDA, or far-right movements, about immigration restriction, German nationalism, anti-Islam positions, opposition to coronavirus restrictions or vaccine mandates, or pro-Russia and anti-NATO sentiment",
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

unknown_events <- germany_acled_df %>%
  filter(event_partisan_type == "unknown") %>%
  filter(!is.na(notes) & str_length(notes) > 20)

cat("Events to classify:", nrow(unknown_events), "\n") # 14229 to class 

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

germany_acled_df <- germany_acled_df %>%
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
germany_acled_df %>%
  count(event_partisan_type_final, classification_source) %>%
  arrange(desc(n)) %>%
  print()

cat("\nRemaining unknowns:", sum(germany_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped4.csv")

text_classified_obs <- germany_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample4.xlsx")

################################################################################

germany_acled_df <- germany_acled_df %>%
  mutate(
    event_partisan_type_final = case_when(
      
      # Querdenken explicitly mentioned → left_and_right
      str_detect(notes, regex("querdenken|lateral thinking", ignore_case = TRUE)) &
        event_partisan_type_final != "left_and_right" ~ "left_and_right",
      
      # Anti-refugee housing → right, not left_and_right
      str_detect(notes, regex("refugee|asylbewerber|flüchtling|asylum seeker", ignore_case = TRUE)) &
        str_detect(notes, regex("against|stop|oppose|verhindern|ablehnung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" ~ "right",
      
      # No-borders / pro-asylum activist groups → left, not left_and_right  
      str_detect(notes, regex("no borders|no one is illegal|deportation|abschiebung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" ~ "left",
      
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      str_detect(notes, regex("querdenken|lateral thinking", ignore_case = TRUE)) &
        classification_source != "original" ~ "rule_correction",
      str_detect(notes, regex("refugee|asylbewerber|flüchtling|asylum seeker", ignore_case = TRUE)) &
        str_detect(notes, regex("against|stop|oppose|verhindern|ablehnung", ignore_case = TRUE)) &
        classification_source != "original" ~ "rule_correction",
      str_detect(notes, regex("no borders|no one is illegal|deportation|abschiebung", ignore_case = TRUE)) &
        classification_source != "original" ~ "rule_correction",
      TRUE ~ classification_source
    )
  )

cat("\nRemaining unknowns:", sum(germany_acled_df$event_partisan_type_final == "unknown"), "\n")

write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped5.csv")

text_classified_obs <- germany_acled_df %>%
  filter(classification_source == "text_classification")
random_sample <- text_classified_obs %>% 
  slice_sample(n=100)
library(writexl)
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample5.xlsx") # LOOKS GREAT. 

###############################################################################################

library(readr)
library(dplyr)
library(stringr)
library(writexl)

germany_acled_df <- read_csv("~/Downloads/germany_acled_partisan_classification_bootstrapped3.csv")

germany_acled_df <- germany_acled_df %>%
  mutate(
    matched_querdenken = (
      str_detect(notes, regex("querdenken|lateral thinking", ignore_case = TRUE)) &
        event_partisan_type_final != "left_and_right" &
        classification_source != "original"
    ),
    matched_anti_refugee = (
      str_detect(notes, regex("refugee|asylbewerber|flüchtling|asylum seeker", ignore_case = TRUE)) &
        str_detect(notes, regex("against|stop|oppose|verhindern|ablehnung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" &
        classification_source != "original"
    ),
    matched_no_borders = (
      str_detect(notes, regex("no borders|no one is illegal|deportation|abschiebung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" &
        classification_source != "original"
    ),
    # Counter-AfD and pro-democracy protests miscoded as right/left_and_right → left
    matched_anti_afd = (
      str_detect(notes, regex(
        "against.*afd|gegen.*afd|protest.*afd|afd.*protest|
         against the rise of far-right|against right-wing extremism|
         sea of lights|lights against the right|
         we are the firewall|firewall.*against.*right|
         remigration|against.*sellner",
        ignore_case = TRUE
      )) &
        event_partisan_type_final %in% c("right", "left_and_right") &
        classification_source != "original"
    ),
    event_partisan_type_final = case_when(
      matched_querdenken   ~ "left_and_right",
      matched_anti_refugee ~ "right",
      matched_no_borders   ~ "left",
      matched_anti_afd     ~ "left",
      TRUE ~ event_partisan_type_final
    ),
    classification_source = case_when(
      matched_querdenken   ~ "rule_correction",
      matched_anti_refugee ~ "rule_correction",
      matched_no_borders   ~ "rule_correction",
      matched_anti_afd     ~ "rule_correction",
      TRUE ~ classification_source
    )
  ) %>%
  select(-matched_querdenken, -matched_anti_refugee, -matched_no_borders, -matched_anti_afd)

# Check how many anti_afd events were flipped
cat("Events flipped by anti_afd rule:", sum(germany_acled_df$classification_source == "rule_correction"), "\n")

# Check discrepancy is gone
discrepancy <- germany_acled_df %>%
  filter(classification_source != "unknown" & event_partisan_type_final == "unknown")
cat("Discrepancy count (should be 0):", nrow(discrepancy), "\n")

table(germany_acled_df$classification_source)
table(germany_acled_df$event_partisan_type_final)

write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped7.csv")

random_sample <- germany_acled_df %>%
  filter(classification_source == "text_classification") %>%
  slice_sample(n = 100)
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample7.xlsx")

####################################################################################

library(readr)
library(dplyr)
library(stringr)
library(writexl)

germany_acled_df <- read_csv("~/Downloads/germany_acled_partisan_classification_bootstrapped3.csv")

germany_acled_df <- germany_acled_df %>%
  mutate(
    
    # -------------------------------------------------------------------------
    # FLAG ROWS MATCHING EACH RULE
    # -------------------------------------------------------------------------
    
    # [R1] Anti-coronavirus measures miscoded as left → right
    # (bootstrapped3 → 4)
    matched_corona_to_right = (
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(notes, regex(
          paste0(
            "coronavirus protection measures|coronavirus lockdown|corona-massnahmen|",
            "impfpflicht|covid-massnahmen|lockdown measures|contact ban|health pass|",
            "querdenken|coronavirus-related measures|compulsory vacc|",
            "corona schutzmasnahmen|against the coronavirus|covid protection|",
            "corona measures|impfung|walk.*corona|corona.*walk"
          ),
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left"
    ),
    
    # [R2] Farmer protests miscoded as left → centre
    # (bootstrapped3 → 4)
    matched_farmers_to_centre = (
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(notes, regex(
          paste0(
            "agrardiesel|agricultural subsid|agrarian polic|farmers.*protest|",
            "bauern|agricultural polic|agrarian package|farmers.*tractor|",
            "tractor.*protest|farmers.*blockade|farmers.*demonstration|",
            "vehicle tax exemption|farmers and craftsmen|farmers.*nationwide|",
            "nationwide farmers"
          ),
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left"
    ),
    
    # [R3] Motorbike/vehicle protests miscoded as left → centre
    # (bootstrapped3 → 4)
    matched_motorbike_to_centre = (
      classification_source %in% c("text_classification", "text_classification_corrected") &
        str_detect(notes, regex(
          paste0(
            "motorbike.*protest|motorcycle.*protest|biker.*protest|ffmc|",
            "bikers.*gathered|motorbike.*ban|weekend.*driving ban|",
            "technical control.*motorbike|motorbike.*technical|",
            "ride free|motor biker"
          ),
          ignore_case = TRUE
        )) &
        event_partisan_type_final == "left"
    ),
    
    # [R4] Anti-refugee housing miscoded as left_and_right → right
    # (bootstrapped5 → 7)
    matched_anti_refugee = (
      str_detect(notes, regex("refugee|asylbewerber|flüchtling|asylum seeker", ignore_case = TRUE)) &
        str_detect(notes, regex("against|stop|oppose|verhindern|ablehnung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" &
        classification_source != "original"
    ),
    
    # [R5] No-borders / pro-asylum / anti-deportation miscoded as left_and_right → left
    # (bootstrapped5 → 7)
    matched_no_borders = (
      str_detect(notes, regex("no borders|no one is illegal|deportation|abschiebung", ignore_case = TRUE)) &
        event_partisan_type_final == "left_and_right" &
        classification_source != "original"
    ),
    
    # [R6] Anti-AfD / pro-democracy miscoded as right or left_and_right → left
    # Extended in v8: catches "Alternative for Germany" spelled out and
    # "in response to a campaign event held by the AfD"
    matched_anti_afd = (
      str_detect(notes, regex(
        paste0(
          "against.*afd|gegen.*afd|protest.*afd|afd.*protest|",
          "against the rise of far-right|against right-wing extremism|",
          "sea of lights|lights against the right|",
          "we are the firewall|firewall.*against.*right|",
          "remigration|against.*sellner|",
          "against.*alternative for germany|alternative for germany.*protest|",
          "response to.*afd|response to a campaign event held by the afd|",
          "against an alternative for germany|against the alternative for germany"
        ),
        ignore_case = TRUE
      )) &
        event_partisan_type_final %in% c("right", "left_and_right") &
        classification_source != "original"
    ),
    
    # [R7] Querdenken / lateral thinking → right
    # Previously mapped to left_and_right in earlier versions;
    # validation consistently codes these as right only.
    matched_querdenken = (
      str_detect(notes, regex("querdenken|lateral thinking", ignore_case = TRUE)) &
        event_partisan_type_final != "right" &
        classification_source != "original"
    )
    
  ) %>%
  
  # -------------------------------------------------------------------------
# APPLY CORRECTIONS (order matters: later rules take precedence)
# -------------------------------------------------------------------------
mutate(
  event_partisan_type_final = case_when(
    matched_corona_to_right     ~ "right",
    matched_farmers_to_centre   ~ "centre",
    matched_motorbike_to_centre ~ "centre",
    matched_anti_refugee        ~ "right",
    matched_no_borders          ~ "left",
    matched_anti_afd            ~ "left",
    matched_querdenken          ~ "right",
    TRUE ~ event_partisan_type_final
  ),
  classification_source = case_when(
    matched_corona_to_right     ~ "rule_correction",
    matched_farmers_to_centre   ~ "rule_correction",
    matched_motorbike_to_centre ~ "rule_correction",
    matched_anti_refugee        ~ "rule_correction",
    matched_no_borders          ~ "rule_correction",
    matched_anti_afd            ~ "rule_correction",
    matched_querdenken          ~ "rule_correction",
    TRUE ~ classification_source
  )
) %>%
  select(-starts_with("matched_"))

# -------------------------------------------------------------------------
# DIAGNOSTICS
# -------------------------------------------------------------------------

cat("\nClassification source breakdown:\n")
print(table(germany_acled_df$classification_source))

cat("\nFinal partisan type breakdown:\n")
print(table(germany_acled_df$event_partisan_type_final))

cat("\nRemaining unknowns:", sum(germany_acled_df$event_partisan_type_final == "unknown"), "\n")

discrepancy <- germany_acled_df %>%
  filter(classification_source != "unknown" & event_partisan_type_final == "unknown")
cat("Discrepancy count (should be 0):", nrow(discrepancy), "\n")

# -------------------------------------------------------------------------
# SAVE & SAMPLE FOR VALIDATION
# -------------------------------------------------------------------------

library(readr)
write_csv(germany_acled_df, "~/Downloads/germany_acled_partisan_classification_bootstrapped8.csv")

random_sample <- germany_acled_df %>%
  filter(classification_source %in% c("rule_correction", "text_classification")) %>%
  group_by(classification_source) %>%
  slice_sample(n = 50) %>%
  ungroup()
write_xlsx(random_sample, "~/Documents/germany_text_class_random_sample8.xlsx")

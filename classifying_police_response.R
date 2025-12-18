################################################################################
# ROBUST POLICE PRESENCE CLASSIFICATION FOR ACLED DATA
# Author: Katelyn Nutley
# Date: 17-12-2025
################################################################################

library(dplyr)
library(readr)
library(stringr)

# Load data
acled_data <- read_csv("~/Documents/GitHub/Police_Response/data/combined/acled_all_countries_combined_classed.csv")

################################################################################
# HELPER FUNCTION: Context-Aware Police Presence Detection
################################################################################

# This has been re-worked repeatedly to improve it; make it more sympathetic to ACLED's data 

detect_police_presence <- function(notes, interaction = NA, actor1 = NA) {
  
  # Return FALSE if no notes
  if (is.na(notes) || nchar(trimws(notes)) < 10) {
    return(FALSE)
  }
  
  notes_lower <- tolower(notes)
  
  # STEP 1: EXCLUDE - Police as the protesters themselves
  police_as_protesters <- c(
    "police officers?.*(gathered|demonstrated|protested|rallied|marched|dropped their)",
    "officers?.*(gathered|demonstrated|protested|rallied).*(in front of|at|outside)",
    "police (union|forces?).*(protested|demonstrated|gathered)",
    "police.*dropped.*(their )?handcuffs",
    "JUSAPOL"
  )
  
  if (any(sapply(police_as_protesters, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(FALSE)
  }
  
  # Check actor field for police as protesters
  if (!is.na(actor1) && grepl("Police Forces", actor1, ignore.case = TRUE)) {
    if (!is.na(interaction) && interaction == "Protesters only") {
      return(FALSE)
    }
  }
  
  # STEP 2: EXCLUDE - Protesting AGAINST police (foreign/past events)
  against_police <- c(
    "protest.*against police (brutality|violence|reform)",
    "demonstrat.*against police",
    "denounced.*police brutality",
    # Foreign events
    "in (solidarity with|response to|view of).*(protest|death).*in (the )?(United States|U\\.S\\.|US|Serbia|Iran|Belarus)",
    "after.*death.*by.*police officer in (the )?(United States|America|US)",
    "black lives matter.*(in the United States|after.*killing.*United States)",
    "police brutality.*(in|after).*(United States|America)"
  )
  
  if (any(sapply(against_police, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(FALSE)
  }
  
  # STEP 3: EXCLUDE - Legislative/administrative mentions of police custody
  custody_legislative <- c(
    "prohibit.*publication.*police custody",
    "amendment.*police custody",
    "law.*would prohibit.*custody",
    "against.*law.*custody orders"
  )
  
  if (any(sapply(custody_legislative, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(FALSE)
  }
  
  # STEP 4: EXCLUDE - Explicit statements of NO police intervention
  no_intervention <- c(
    "no report of.*police intervention",
    "no police intervention",
    "without.*police intervention",
    "police did not intervene"
  )
  
  if (any(sapply(no_intervention, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(FALSE)
  }
  
  # STEP 5: DETECT - Strong evidence of police intervention
  strong_intervention <- c(
    "police (intervened|intervention|dispersed|dispersing)",
    "police.*removed.*demonstrators",
    "police.*used (tear gas|water cannon|pepper spray)",
    "officers? (intervened|dispersed|removed|deployed)",
    "(tear gas|water cannon|pepper spray) (was|were) used",
    "police.*physically.*(remove|removed)",
    "police.*blocked.*(access|entrance|road|demonstrators)",
    "police.*arrested.*(demonstrators|protesters)",
    "police.*detained.*(protesters|demonstrators)",
    "\\d+.*arrested (by police|on site)",
    "scuffles?.*(with|between).*police",
    "clashed with police",
    "police charged.*protesters",
    "baton charge",
    "police cordon",
    "kettled|kettling",
    "police formed.*cordon"
  )
  
  if (any(sapply(strong_intervention, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(TRUE)
  }
  
  # STEP 6: DETECT - Interaction field indicates state forces
  if (!is.na(interaction) && grepl("State forces", interaction, ignore.case = TRUE)) {
    return(TRUE)
  }
  
  # STEP 7: DETECT - Weaker evidence (only if no exclusions triggered)
  weak_presence <- c(
    "police presence",
    "police (were|was) present",
    "escorted by police",
    "police escort",
    "police monitored",
    "police observed",
    "riot police.*deployed"
  )
  
  if (any(sapply(weak_presence, function(p) grepl(p, notes_lower, perl = TRUE)))) {
    return(TRUE)
  }
  
  # Default: no police presence
  return(FALSE)
}

################################################################################
# CLASSIFY ALL EVENTS
################################################################################

acled_data <- acled_data %>%
  rowwise() %>%
  mutate(
    police_presence = detect_police_presence(notes, interaction, actor1)
  ) %>%
  ungroup()

table(acled_data$police_presence)

################################################################################
# SUMMARY STATISTICS
################################################################################

# By country
country_summary <- acled_data %>%
  group_by(country) %>%
  summarise(
    total = n(),
    with_police = sum(police_presence),
    pct = round(with_police / total * 100, 2)
  ) %>%
  arrange(desc(pct))

print(country_summary)

################################################################################
# VALIDATION SAMPLE
################################################################################

set.seed(123)
validation_sample <- acled_data %>%
  filter(police_presence == TRUE) %>%
  sample_n(20) %>%
  select(country, event_date, interaction, notes)

# Hand coded everything; very pleased with this! 

################################################################################
# EXPORT POLICE PRESENCE SUBSET
################################################################################

police_response_subset <- acled_data %>%
  filter(police_presence == TRUE)

write_csv(police_response_subset, "~/Downloads/acled_police_response_subset.csv")
write_csv(acled_data, "~/Documents/GitHub/Police_Response/data/combined/acled_classified_police_presence.csv")

################################################################################
# DETAILED BREAKDOWN FOR PAPER
################################################################################

# Event types with police presence
event_type_summary <- acled_data %>%
  filter(police_presence == TRUE) %>%
  count(event_type, sub_event_type, sort = TRUE) %>%
  head(10)

print(event_type_summary)

# Temporal trends
temporal_summary <- acled_data %>%
  filter(police_presence == TRUE) %>%
  group_by(year) %>%
  summarise(
    events_with_police = n(),
    total_events_that_year = sum(acled_data$year == year[1])
  ) %>%
  mutate(
    pct = round(events_with_police / total_events_that_year * 100, 2)
  ) %>%
  arrange(year)

print(temporal_summary)

# Cross-tabulation: interaction type vs police presence
interaction_table <- acled_data %>%
  count(interaction, police_presence) %>%
  tidyr::pivot_wider(names_from = police_presence, values_from = n, values_fill = 0)

print(interaction_table)

###############

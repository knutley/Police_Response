## L-R Machine Learning on Protest Data 
## Code by: Katie Nutley 
## 01-05-2025
## Data from: ACLED 

# 1. Load Libraries

# Load libraries
library(dplyr)
library(SnowballC)
library(stringdist)  # For fuzzy string matching, paired with widyr. 
library(stringr)
library(text2vec)
library(textclean)
library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)
library(widyr) # This is new; for calculating pairwise functions. 

# 2. Load in Data 
df <- read.csv('~/Downloads/1997-01-01-2025-05-01-Europe-United_Kingdom.csv')

# Basic data exploration... 
cat("Number of events:", nrow(df), "\n")
cat("Time period:", min(df$event_date), "to", max(df$event_date), "\n") 
# Need to make a note of this, because I actually pulled it in May 2025; so the 
# data upload seems to be considerably lagged. 
cat("Number of unique primary actors:", length(unique(df$assoc_actor_1)), "\n")
unique(df$assoc_actor_1) # There seem to be duplicates here, so I will need to 
# pre-process and standardise this. 
cat("Number of unique secondary actors:", length(unique(df$assoc_actor_2)), "\n")
unique(df$assoc_actor_2) # There are fewer entries here, but maybe it could be 
# used to reinforce the model? Idk, I'll see where this goes. 

# Just to be clear, I've decided to train the model on associated actors because 
# the 'actors1' and 'actors2' columns are more descriptive, i.e. the delineate
# whether it was a religious group, women, a labour group, etc. Just not much 
# to train on or classify. 

# 3. Actor Preprocessing and Standardisation
# First, we need to standardise actor names
standardise_actors <- function(actor_vector) {
  # Convert to character
  actors <- as.character(actor_vector)
  
  # Replace NAs with empty string
  actors[is.na(actors)] <- ""
  
  # Lowercase everything
  actors <- tolower(actors)
  
  # Remove common prefixes/suffixes that don't really change meaning
  actors <- str_replace_all(actors, "^the\\s+|\\s+the\\s+|\\s+group$|\\s+movement$", " ")
  
  # Remove punctuation and extra spaces
  actors <- str_replace_all(actors, "[[:punct:]]", " ")
  actors <- str_replace_all(actors, "\\s+", " ")
  actors <- str_trim(actors)
  
  return(actors)
}

# Apply Standardisation to the existing df
df <- df %>%
  mutate(
    actor1_std = standardise_actors(assoc_actor_1),
    actor2_std = standardise_actors(assoc_actor_2)
  )

View(df) # Right, this seems to have worked. The first few entries seemed a bit
# sparse though. I wonder how many rows are actually filled in? Would it be enough
# to train this model? :/ 
sum(df$assoc_actor_1 == "" | is.na(df$assoc_actor_1)) # Okay, so basically we 
# have 6000 entries that actually exist. I wonder if that's enough statistical power?
# I guess I could just go ahead do the LDA and then do topic coherence checks after? 

# 4. Create Political Lexicons for Classification
# These are keyword dictionaries that help identify political orientation; I plan on
# shopping these terms around to some qualitative/critical scholars to see if there's
# anything they might disagree with. 
leftist_keywords <- c(
  "socialist", "communist", "labour", "workers", "union", "leftist", "marxist", 
  "antifa", "anti-fascist", "anarchist", "progressive", "climate", "environmental", 
  "extinction rebellion", "lgbt", "lgbtq", "trans", "feminist", "blm", "black lives matter", 
  "social justice","anti-capitalist", "occupy", "activism", "green", "greenpeace", "migrant", 
  "refugee", "solidarity", "anticapitalist", "anti racist", "antiracist", "anti war",
  "antiwar", "woke", "pro-choice", "queer", "pro-palestine", "pro palestine", "liberal"
)  

rightwing_keywords <- c(
  "nationalist", "conservative", "tory", "traditionalist", "patriot", "libertarian",
  "far-right", "alt-right", "anti-immigration", "brexiteer", "brexit", "ukip", "bnp",
  "anti-eu", "anti-islam", "anti-muslim", "anti-lgbtq", "religious conservative", 
  "anti-woke", "pro-life", "identitarian", "nationalist", "pro-britain", "patriotic",
  "anti-refugee", "anti-migrant", "ethnonationalist", "white nationalist", "edl",
  "tommy robinson", "national front", "britain first", "reform", "capitalist"
) # Also ask here. I asked my mother who is conservative (albeit American), what
# 

# 5. Text Cleaning and Processing
# Cleaning and standardising the notes field
df_clean <- df %>% 
  mutate(
    notes = as.character(notes),
    notes_clean = replace_contraction(notes),
    notes_clean = str_replace_all(notes_clean, "[^[:alnum:] ]", " "),
    notes_clean = tolower(notes_clean),
    notes_clean = str_trim(str_squish(notes_clean))
  )

# Tokenise text
tidy_notes <- df_clean %>% 
  unnest_tokens(word, notes_clean) %>%
  anti_join(stop_words, by = "word") %>%
  # Filter out numbers
  filter(!str_detect(word, "^[0-9]+$")) %>%
  # Stem words
  mutate(word = wordStem(word))

# 6. Advanced Actor Classification

# 6.1 Direct keyword matching for actors
classify_actor <- function(actor, leftist_keys, rightwing_keys) {
  actor <- tolower(actor)
  
  if (actor == "" || is.na(actor)) {
    return("Unclassified")
  }
  
  # Check for exact matches in actor names
  for (key in leftist_keys) {
    if (str_detect(actor, paste0("\\b", key, "\\b"))) {
      return("Left")
    }
  }
  
  for (key in rightwing_keys) {
    if (str_detect(actor, paste0("\\b", key, "\\b"))) {
      return("Right")
    }
  }
  
  return("Unclassified")
}

# Apply classification to actors
df_classified <- df_clean %>%
  mutate(
    actor1_ideology = sapply(actor1_std, classify_actor, leftist_keywords, rightwing_keywords),
    actor2_ideology = sapply(actor2_std, classify_actor, leftist_keywords, rightwing_keywords)
  )
 
table(df_classified$actor1_ideology) 
table(df_classified$actor2_ideology) # This picked absolutely no right-wing protestors
# is that right? *** TO DO: You need to come back to this and comb through some of these
# secondary associated actors and see if you might potentially be missing something. 
# To begin with it might be worth running a unique(df_classified$assoc_actor_2) and 
# then going through that by hand and adding descriptors to leftist_keywords or 
# rightwing_keywords. 

# 7. Topic Modeling for Unclassified Events
# Filter unclassified events for topic modeling
unclassified_events <- df_classified %>%
  filter(actor1_ideology == "Unclassified" & actor2_ideology == "Unclassified")

if (nrow(unclassified_events) > 10) {  # Only proceeding if I have enough data
  # Create Document-Term Matrix for unclassified events
  # First, ensure tidy_notes has the event_id_cnty column
  # We need to add the event_id_cnty to tidy_notes from df_clean
  tidy_notes_with_id <- df_clean %>% 
    mutate(row_id = row_number()) %>%  # Add a row identifier
    select(row_id, event_id_cnty, notes_clean) %>%  # Keeping only necessary columns
    unnest_tokens(word, notes_clean) %>%
    anti_join(stop_words, by = "word") %>%
    filter(!str_detect(word, "^[0-9]+$")) %>%
    mutate(word = wordStem(word))
  
  # Now joining with unclassified_events
  unclass_notes <- tidy_notes_with_id %>%
    inner_join(unclassified_events %>% select(event_id_cnty), by = "event_id_cnty")
  
  # Create DTM for unclassified events
  unclass_dtm <- unclass_notes %>%
    count(event_id_cnty, word) %>%
    cast_dtm(document = event_id_cnty, term = word, value = n)
  
  # Apply LDA topic modeling if we have enough terms
  if (dim(unclass_dtm)[2] > 5) {
    # LDA Topic Modeling with 3 topics
    lda_model <- LDA(unclass_dtm, k = 3, control = list(seed = 1234))
    
    # Extract topics
    topics <- tidy(lda_model, matrix = "gamma")
    doc_topics <- topics %>%
      group_by(document) %>%
      top_n(1, gamma) %>%
      ungroup()
    
    # Get top terms for each topic
    top_terms <- tidy(lda_model, matrix = "beta") %>%
      group_by(topic) %>%
      top_n(15, beta) %>%
      arrange(topic, -beta)
    
    # Print top terms to help determine topic ideology
    print("Top terms by topic:")
    print(top_terms)
  }
}

# 8. Context-Based Classification
# Createing a function that looks at context words around actors
context_classify <- function(text, left_keys, right_keys) {
  text <- tolower(text)
  
  # Count matches for each ideology
  left_matches <- sum(sapply(left_keys, function(x) str_count(text, paste0("\\b", x, "\\b"))))
  right_matches <- sum(sapply(right_keys, function(x) str_count(text, paste0("\\b", x, "\\b"))))
  
  if (left_matches > right_matches) {
    return("Left")
  } else if (right_matches > left_matches) {
    return("Right")
  } else {
    return("Unclassified")
  }
}

# Apply context-based classification
df_classified <- df_classified %>%
  mutate(
    context_ideology = sapply(notes_clean, context_classify, leftist_keywords, rightwing_keywords)
  ) # This has proven to be really important! 

# 9. Combined Classification Strategy
df_classified <- df_classified %>%
  mutate(
    final_ideology = case_when(
      actor1_ideology != "Unclassified" ~ actor1_ideology,
      actor2_ideology != "Unclassified" ~ actor2_ideology,
      context_ideology != "Unclassified" ~ context_ideology,
      TRUE ~ "Unclassified"
    )
  )

# 10. Analyse Results
ideology_summary <- df_classified %>%
  count(final_ideology) %>%
  mutate(percentage = n / sum(n) * 100)

print("Ideology Classification Summary:")
print(ideology_summary) # This is better than my first iteration, which
# of the possibly classifiable associated actors had 16% missing. This iteration,
# which is my third or fourth has 15% missing. I need to go back and comb through
# the assoc_actor_1 and 2 to see if there's anything in there that I'm obviously
# missing, but this is kind of busy work and time is presently tight. 

# trble_shoot_text <- df_classified %>% # This is for doing the above, of course. 
#  filter(final_ideology == "Unclassified") %>%
#  select(notes)
# View(trble_shoot_text)

# Sample events for each category
sample_events <- df_classified %>%
  group_by(final_ideology) %>%
  slice_sample(n = 3) %>%
  ungroup() %>%
  select(event_date, assoc_actor_1, assoc_actor_2, notes, final_ideology)

print("Sample events by category:")
print(sample_events)

# 11. Validation Check - Manual review sample
set.seed(123)
validation_sample <- df_classified %>%
  filter(final_ideology != "Unclassified") %>%
  sample_n(min(20, nrow(.))) %>%
  select(event_date, assoc_actor_1, assoc_actor_2, notes, final_ideology)

print("Validation sample for manual review:")
print(validation_sample)
view(validation_sample) # Went through this all by hand and it looked great. 

# 12. Export results
write.csv(df_classified, "acled_uk_classified.csv", row.names = FALSE)

#################################################################################
# Katie, you need to rework this visualisation, because it changed after you 
# altered something upstream. Whoops. 

# Create a summary visualization of protest ideologies over time
df_classified %>%
  mutate(year = lubridate::year(as.Date(event_date))) %>%
  group_by(year, final_ideology) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = year, y = count, fill = final_ideology)) +
  geom_bar(stat = "identity", position = "stack") +
  theme_minimal() +
  labs(title = "UK Protests by Political Ideology (1997-2025)",
       x = "Year", 
       y = "Number of Events",
       fill = "Ideology") +
  theme(legend.position = "bottom")

# Combining All Countries - Post Partisan Classification
# Author: Katie Nutley
# Date: 17-12-2025

# Load Libs 

library(dplyr)
library(tidyverse)
library(readr)

# Read in country-level data: 
uk_data <- read_csv("~/Documents/GitHub/Police_Response/data/United Kingdom/uk_acled_partisan_classification.csv")
france_data <- read_csv("~/Documents/GitHub/Police_Response/data/France/france_acled_partisan_classification.csv")
germany_data <- read_csv("~/Documents/GitHub/Police_Response/data/Germany/germany_acled_partisan_classification.csv")
italy_data <- read_csv("~/Documents/GitHub/Police_Response/data/Italy/italy_acled_partisan_classification.csv")
spain_data <- read_csv("~/Documents/GitHub/Police_Response/data/Spain/spain_acled_partisan_classification.csv")

# Combine: 
combined_data <- rbind(uk_data, france_data, germany_data, italy_data, spain_data)

# Write and Upload in GitHub desktop: 
write_csv(combined_data, "~/Documents/GitHub/Police_Response/data/combined/acled_all_countries_combined_classed.csv")

table(uk_data$event_partisan_type)
table(france_data$event_partisan_type)
table(germany_data$event_partisan_type)
table(italy_data$event_partisan_type)
table(spain_data$event_partisan_type)


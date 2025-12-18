# BERT CLASSIFICATION FOR POLICE RESPONSE SEVERITY
# Author: Katie Nutley 
# Date: 18-12-2025 

# Revised the last point on the 18th; but relied on a previous script: train_model.py

library(readr)
library(dplyr)
library(writexl)
library(readxl)

# Upload the police presence subset from binary classification; random subsample frac=0.1

police_df <- read_csv("~/Downloads/acled_police_response_subset.csv")
set.seed(123)
severity_sample <- acled_data %>%
  filter(police_presence == TRUE) %>%
  sample_frac(0.1) %>%
  select(country, event_date, interaction, notes)

# Going to add columns so that you can hand code this; then save and go hand code it

severity_sample$presence <- "0"
severity_sample$coercion <- "0"
severity_sample$arrest <- "0"
severity_sample$brutality <- "0"
write_xlsx(severity_sample, "~/Documents/GitHub/Police_Response/severity_sample.xlsx")

# Pull this over to your python script

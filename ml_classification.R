### Machine-Learning Document Classification ###
## 10.06.2023 ## 

##########################
## Part I: Read in Data ##
##########################

library(readr)
raw_data <- read_csv('/Users/katienutley/Downloads/1997-01-01-2022-11-09-Europe-United_Kingdom.csv')

# This data has already been filtered according to whether there was a police 
# interaction of some kind. Interaction and police response (i.e. coercion, brutalisation,
# arrest, and death) are not mutually exclusive. In short, you can have police 
# interaction without these four things. Additionally, it should be noted that 
# this dataset includes all countries in GB, so I need to exclude based on column
# 'admin1'. 

# Exclude Unnecessary Observations # 

eng_data <- raw_data[-c(4207:5563),] #Just quickly deleted all observations that
# were from Northern Ireland, Scotland or Wales. 
View(eng_data)

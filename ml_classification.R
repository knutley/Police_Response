### Machine-Learning Text Classification ###
## 10.06.2023 ## 

##########################
## Part I: Read in Data ##
##########################

library(readr)
raw_data <- read_csv('/Users/katienutley/Downloads/1997-01-01-2022-11-09-Europe-United_Kingdom.csv')
View(raw_data)

# This data has already been filtered according to whether there was a police 
# interaction of some kind. Interaction and police response (i.e. coercion, brutalisation,
# arrest, and death) are not mutually exclusive. In short, you can have police 
# interaction without these four things. Additionally, it should be noted that 
# this dataset includes all countries in GB, so I need to exclude based on column
# 'admin1'. 

# Exclude Unnecessary Observations # 

countries_to_delete <- c("Northern Ireland", "Scotland", "Wales")
eng_data <- subset(raw_data, !(admin1 %in% countries_to_delete))
View(eng_data)

######################################################
## Part II: Machine Learning Classification of Text ##
######################################################

# Using GitHub/PostdOK's script for Naive Bayes binary classification:
# https://github.com/PostdOK/automated-text-classification-algorithms/blob/master/script_nb_binary.R

# Load Libraries # 

library(quanteda)
library(quanteda.textmodels)
library(caret)

# Create Training Data from Dataset # 

# I am going to be dealing with police interaction as the first binary
# classification, so I need to collect training documents that are indicative
# of police interaction. 

# These excerpts are taken from the Notes section of the dataset. I went through
# the first 50 observations I came across to source these. Obviously, if this works
# I would then go through a larger number of texts to make the algorithm more accurate. 

training_text <- c(
  "verbal tension with the police",
  "tension with the police",
  "police forces dispatched",
  "forces dispatched to the scene",
  "police dispatches",
  "police arrested",
  "police arrested a number of protestors",
  "protestors were arrested",
  "police forces arrested",
  "demostrators were arrested",
  "arrested demonstrators",
  "protestors were shot", 
  "protestor was shot", 
  "demostrators were shot", 
  "demonstrator was shot", 
  "protestors were killed", 
  "protestor was killed", 
  "demonstrators were killed", 
  "demonstrator was killed" ,
  "voice their anger",
  "gathered outside ",
  "to occupy",
  "staged a protest",
  "call for the government",
  "marched through",
  "striking workers",
  "occupied", 
  "glued themselves",
  "blocked off",
  "gathered at",
  "death of Mahsa",
  "justice for Chris Kaba",
  "Chris Kaba was shot by police",
  "Flight 752",
  "marched from",
  "picketed outside",
  "show their support",
  "union representative",
  
)

training_labels <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, )

training_data <- data.frame(training_labels=training_labels, 
                            training_text=training_text)
View(training_data)

# Convert Training Data into a Corpus # 

training_corpus <- corpus(training_data,text_field = "training_text") # This threw up an error message that 
# reads, "Error in corpus.data.frame(training_data): text_field column not found 
# or invalid." 

#library(tm)

#training_corpus <- Corpus(VectorSource(training_data$training_text)) 
# This appears to have worked? There was no error or warning? 

# The author of the seed script I used said, "different to other algorithms, I 
# recommend to split the data before pre-processing set seed if necessary and draw
# the ids for training and test data." I do not quite understand what this means yet,
# but I hope it becomes apparent as I go on. 

set.seed(123)
id_train <- sample(1:nrow(training_data), round(0.75*nrow(training_data),0), replace = FALSE)

# Add IDs for Corpus # 

docvars(training_corpus, "id_numeric") <- 1:ndoc(training_corpus) # This just threw
# up an error message that reads, "Error: ndoc() only works on corpus, dfm, readtext,
# spacyr_textmodel, tokens objects." 
class(training_corpus) # This identifies the training_corpus as a corpus, though?
# You need to ask Hauke about this - have tried handling this every which way. 
# Maybe it's unnecessary for such a small corpus that only has 1 variable and 1 
# text column? So, 

# Draw the Samples # 

# In this case words are stemmed and converted to lower case. Punctuation and numbers
# are removed. Using quanteda, this can be done while creating a document feature 
# matrix, which is needed to run the algorithms. 

training_dtm <- corpus_subset(training_corpus, id_numeric %in% id_train) %>%
  tokens() %>% 
  dfm() %>% 
  dfm_wordstem()
# Threw up the following error message: "Error: corpus_subset() only works on 
# corpus objects." 







TestDTM<-corpus_subset(training_corpus,!id_numeric %in% id_train) %>%
  tokens() %>% 
  dfm()

#Train the model and make predictions
#
#Train the model to classify into your coding variable (in this case called "VAR")
nbModel<-textmodel_nb(training_dtm,docvars(training_dtm,"training_labels"))
#Use the model to classify the test data. dfm_match necessary to have the same features for both document-feature matrixes.
predicted_class <- predict(nbModel, newdata = dfm_match(TestDTM, features = featnames(training_dtm)))

#Evaluate the model
#
#Get real values
actual_class <- docvars(TestDTM, "training_labels")
#Create confusion matrix
cm<-confusionMatrix(table(actual_class, predicted_class),mode = "everything")














# Sample Test Data # 

# Asked a random number generator to choose 3 values between 2 and 4206. 
# It chose: 1175, 4033, and 3096; I will take the notes section
# from these 3 rows to use as test texts. 

View(eng_data)
test_text <- c("On 27 December 2021, around 30 animal rights activists gathered 
               in Abbots Bromley to protest against a mock fox hunt, 
               known as a drag hunt, scheduled to take place nearby.", 
               "On 22 August 2020, activists armed with megaphones, 
               banners and an inflatable canoe marched through Norwich city 
               centre to show their support for refugees braving 
               'torrid conditions' to cross the channel. 
               A Green Party city councillor also joined the march.",
               "On 13 February 2020, dozens of Extinction Rebellion activists 
               protested outside of the council offices in Kendal to oppose 
               West Cumbria Mining's plans for the UK's first new deep coal 
               mine in 30 years, set to be off a seabed.")

# Convert Training and Test Data into a DTM # 




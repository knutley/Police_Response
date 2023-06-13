### Spatio-Temporal Models of Police Response in England ### 
## 03.03.2023 ## 

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

####################################################################
## Part II: Police Response Classification Using Machine Learning ##
####################################################################

# This is just a test to make sure that you've got the code right, 
# once you figure it out, you can add more keywords and different police
# response classifications. Also, does this class as Machine Learning? I feel like 
# it doesn't? 

# Load Library # 

library(dplyr)

# Define the Keywords and Corresponding Response Classification #

general_response_kw <- c("arrest", "beat", "brutal", "coerce", "coercion", "detain", 
                         "harm", "hurt", "injure", "injury", "kill", "killed")
general_response_classes <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) 

# Again, you will have to go through the Notes section and find keywords that are 
# associated with police response. This also needs to be done for each of your 
# typologised police responses - arrest, brutalisation, coercion, and death. 

# Classification Based on Your Keywords # 

classify_general_response_kw <- function(text) {
  for (i in 1:length(general_response_kw)) {
    if (grepl(general_response_kw[i], text, ignore.case = TRUE)) {
      return(general_response_classes[i])
    }
  }
  return("0")  # Assign 0 if none of your keywords match
}

# Apply Classification # 

general_response_data <- mutate(eng_data, classification = sapply(eng_data$notes,
                                                          classify_general_response_kw))

View(general_response_data) # Having looked at the Notes column, this appears to 
# have worked. 

# Right, this works for the time being, but I feel it's rather unsophisticated?
# I think I should be using a NB classifier? ** LOOK INTO THIS **

# Possible helpful texts: 
# 'Text Classification and Naive Bayes' (McCallum and Nigam, 1998)
# 'Bag of Tricks for Efficient Text Classification' (Joulin et al., 2017)
# Maybe consider a dictionary approach here? (Muddiman et al., 2019)
# Also, have a look at Ken Benoit's text classification models. Feel like I've 
# come across one in particular in class reading that might be useful. 

#############################################################################
## Part III: Building a Predictive Spatio-Temporal Model of Police Response #
#############################################################################

# Install and Load Libraries # 

install.packages("sf") # Had an issue here downloading sf. Trying this instead:

install.packages("remotes")
remotes::install_github("r-spatial/sf") # This still hasn't worked! Maybe
# running rgdal and rgeos packages will work because they're technically 
# dependencies for the sf package? 

install.packages("rgdal") # This apparently didn't work? 
install.packages("rgeos") # But, this did? 

# Tried runnning install.packages on sf again and it just didn't work. Ask someone
# how I go about doing this? 

# Okay, so on Hauke's advice, when I tried to download this in the office this 
# morning, it worked. It appears to have something to do with my internet bandwidth
# at home. - Just a note for the future. 

library(sf)
library(sp)
library(spBayes)
library(spdep)
library(spTimer)

# Convert the Date Column to the Proper Format # 

view(general_response_data)
general_response_data$event_date <- as.Date(general_response_data$event_date,
                                            format = "%d/%m/%Y")
temporal_data <- general_response_data$event_date

# Defining the Binary Variable of Interest # 

police_response <- general_response_data$classification

# Create a Spatial Dataframe with Lat/Long Values # 

# Looking at the documentation for the sp package; p. 88 of: 
# https://cran.r-project.org/web/packages/sp/sp.pdf

coordinates <- data.frame(general_response_data$longitude, general_response_data$latitude)
colnames(coordinates) <- c("longitude", "latitude")
spatial_data <- SpatialPointsDataFrame(coordinates, general_response_data)
View(spatial_data) # This appears to have worked, but created a little list
# with two factors within the one coordinates column, which I feel like could be 
# tricky later? 

# The Spatio-Temporal Model Formula # 

st_formula <- police_response ~ spatial_data + temporal_data

# Fit the Spatio-Temporal Model # 

st_model <- spGLM(st_formula, data = spatial_data, family = "gaussian")

# Right, so I'm getting an error message here that reads: "Error in model.frame.
# default(formula = formula, data = data, drop.unused.levels = TRUE) : invalud
# type (S4) for variable 'spatial_data'". **LOOK INTO THIS**

# Perform Prediction Using Model #

st_prediction <- predict(model, newdata = spatial_data)

# So, I think this is what predictive aspect of the model would look like, but, 
# until I can figure out what the issue with the fit on the spatio-temporal model
# is, I can't run it. 

###############################
# Part IV: Data Visualisation # 
###############################

# Data Vis. for the Predictive Model # 

# I've never done this, so I'm kind of unsure what this should look like - would 
# it simply be a column with predictive scores? 

# Data Vis. for the Predictive Scores # 

# I'd like to effectively overlay the predictive scores onto a shapefile (unsure
# if that's the right word), so that you can get the county-specific predictive 
# scores of police response and fill with a gradient according to severity. 

# Found this example code from r-spatial.github: 

# columbus <- st_read(system.file("shapes/columbus.shp", package="spData")[1], quiet=TRUE)
# col.gal.nb <- read.gal(system.file("weights/columbus.gal", package="spData")[1])
# coords <- st_coordinates(st_centroid(st_geometry(columbus)))
# xx <- poly2nb(as(columbus, "Spatial"))
# dxx <- diffnb(xx, col.gal.nb)
# plot(st_geometry(columbus), border="grey")
# plot(col.gal.nb, coords, add=TRUE)
# plot(dxx, coords, add=TRUE, col="red")
# title(main=paste("Differences (red) in Columbus GAL weights (black)",
                 "and polygon generated queen weights", sep="\n"), cex.main=0.6)

england <- st_read('/Users/katienutley/Downloads/Great_Britain_shapefile/gb_10km.shp')
coords <- st_coordinates(st_centroid(st_geometry(england)))
neighbours <- poly2nb(as(england, "Spatial")) # Right, but this isn't right
# either because I want the coordinates from the original dataset overlaid on
# top of a shapefile? 




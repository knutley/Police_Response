# Control Variable Construction
# Author: Katelyn Nutley
# Date: 08-06-2026

# Load required libraries: 
library(readr)
library(dplyr)
library(stringr)
library(lubridate)
library(sf)
library(giscoR)
library(osmdata)
library(osmextract)
library(eurostat)
library(httr2)
library(purrr)

options(timeout = 600) # you'll need this for the OSM stuff later 

# =============================================================================
# DATA LOADING AND MERGING
# =============================================================================

acled <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_classified_severity1.csv")
big_acled <- read_csv("~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_classified_police_presence1.csv")

table(acled$police_presence...8) # just checking to make sure everything looks right 
table(acled$arrest)
table(acled$brutality)

# Check what columns they share
intersect(names(acled), names(big_acled))

# Check for a unique identifier to join on
head(acled$event_id_cnty)
head(big_acled$event_id_cnty)

acled_merged <- big_acled %>%
  left_join(
    acled %>% select(event_id_cnty, setdiff(names(acled), names(big_acled))),
    by = "event_id_cnty"
  )

# Check if shared columns match between the two
shared_cols <- intersect(names(acled), names(big_acled))

# Check for a unique identifier to join on
head(acled$event_id_cnty)
head(big_acled$event_id_cnty)

acled_merged <- big_acled %>%
  left_join(
    acled %>% select(event_id_cnty, police_presence...7, police_presence...8,
                     setdiff(names(acled), names(big_acled))),
    by = "event_id_cnty"
  )

nrow(acled_merged)  # should be 109151
table(acled_merged$event_partisan_type_final)

# =============================================================================
# 1. PROTEST SIZE
# =============================================================================

sort(table(acled_merged$tags), decreasing = TRUE) # okay, this kind of gave me a view

# First Pass (this is going to take a few because it's so unstandardised)
acled_merged <- acled_merged %>%
  mutate(
    # Extract the first number mentioned in the tag
    crowd_size_raw = str_extract(tags, "[0-9,]+") %>% 
      str_remove_all(",") %>% 
      as.numeric(),
    
    # Bin into categories
    crowd_size_cat = case_when(
      is.na(tags) | str_detect(tags, "no report") ~ "No report",
      str_detect(tags, "handful|a few|several|some|small group|a dozen$|dozens$") ~ "Very small (<20)",
      crowd_size_raw < 20                          ~ "Very small (<20)",
      crowd_size_raw < 50                          ~ "Small (20-49)",
      crowd_size_raw < 100                         ~ "Medium (50-99)",
      crowd_size_raw < 500                         ~ "Large (100-499)",
      crowd_size_raw < 1000                        ~ "Very large (500-999)",
      crowd_size_raw < 5000                        ~ "Mass (1,000-4,999)",
      crowd_size_raw >= 5000                       ~ "Massive (5,000+)",
      str_detect(tags, "hundreds$|several hundred") ~ "Large (100-499)",
      str_detect(tags, "thousands|tens of thousands") ~ "Massive (5,000+)",
      TRUE                                         ~ "Unknown"
    )
  )

table(acled_merged$crowd_size_cat) # 1459 unknowns; let's fix that... 

acled_merged %>% 
  filter(crowd_size_cat == "Unknown") %>% 
  count(tags, sort = TRUE) %>%
  print(n = 30) # okay, so they're all word based 

# Second Pass; fixing word based numbers 
acled_merged <- acled_merged %>%
  mutate(
    crowd_size_cat = case_when(
      # Fix previously unknown word-based numbers
      str_detect(tags, "one|two|three|four|five|six|seven|eight|nine|ten$") ~ "Very small (<20)",
      str_detect(tags, "a dozen|one dozen|around a dozen|around one dozen|about a dozen|a few dozen|two dozen|around two dozen") ~ "Very small (<20)",
      str_detect(tags, "dozens$|several dozen|some dozen|a few dozens|several dozens") ~ "Small (20-49)",
      str_detect(tags, "dozens of vehicles|dozens of tractors|hundreds of vehicles|hundreds of tractors") ~ NA_character_,
      str_detect(tags, "scores") ~ "Small (20-49)",
      str_detect(tags, "tens$|around thirty|around forty|around fifty") ~ "Small (20-49)",
      str_detect(tags, "a hundred|around a hundred|nearly a hundred|about a hundred|more than a hundred|around one hundred") ~ "Large (100-499)",
      str_detect(tags, "half a thousand|more than a thousand|around a thousand|nearly a thousand|a thousand") ~ "Mass (1,000-4,999)",
      str_detect(tags, "^crowd size=large$|a large group|a large number|a large crowd|massive|numerous|many") ~ "Large (100-499)",
      str_detect(tags, "^crowd size=small$|a group$|a small group|a small number") ~ "Very small (<20)",
      str_detect(tags, "no size") ~ "No report",
      TRUE ~ crowd_size_cat  # keep existing classifications
    )
  )

table(acled_merged$crowd_size_cat)
acled_merged %>% 
  filter(crowd_size_cat == "Unknown") %>% 
  count(tags, sort = TRUE)

# Final round of manual fixes - mostly due to fatigue 
acled_merged <- acled_merged %>%
  mutate(
    crowd_size_cat = case_when(
      str_detect(tags, "^crowd size=dozen$|^crowd size=a good dozen|^crowd size=around fifteen|^crowd size=fifteen|^crowd size=about fifteen|^crowd size=twelve") ~ "Very small (<20)",
      str_detect(tags, "about fifty|around twenty|^crowd size=about twenty|^crowd size=about forty|^crowd size=about thirty") ~ "Small (20-49)",
      str_detect(tags, "^crowd size=more than fifty") ~ "Medium (50-99)",
      str_detect(tags, "few hundred|hundred")               ~ "Large (100-499)",
      str_detect(tags, "crowds|a crowd|a number of|a low number") ~ "Large (100-499)",
      str_detect(tags, "crowd size=thousand")  ~ "Mass (1,000-4,999)",
      str_detect(tags, "no reporrt|no reoport|report$") ~ "No report",
      TRUE ~ crowd_size_cat
    )
  )

table(acled_merged$crowd_size_cat) # excellent, only 89 unknowns now

# =============================================================================
# 2. PROTESTOR VIOLENCE
# =============================================================================

# Using pattern matching to assign violence to protestors (this is based on qualitative 
# reading of the notes field entries)

acled_merged <- acled_merged %>%
  mutate(
    protestor_violence = case_when(
      str_detect(tolower(notes),
                 "threw (stones?|rocks?|bottles?|fireworks?|firecrackers?|molotov|projectiles?|paint)|
        |hurled (stones?|rocks?|bottles?|fireworks?|firecrackers?|molotov|projectiles?)|
        |threw .{0,20} at (police|officers|gendarm)|
        |hurled .{0,20} at (police|officers|gendarm)|
        |clashed with police|clashed with officers?|clashed with gendarm|
        |attacked (police|officers|gendarm)|
        |assaulted (police|officers|gendarm)|
        |molotov|broke through (police|a police|the police)|
        |stormed (the parliament|the building|the prefecture|the courthouse)|
        |set fire to (police|vehicles|cars|a police)|
        |vandali[sz]ed") ~ 1,
      TRUE ~ 0
    )
  )

table(acled_merged$protestor_violence) # yeah, that seems right - protests not riot 

# How much overlap with ACLED's own coding?
table(acled_merged$protestor_violence, acled_merged$sub_event_type)

# Flagged violent but ACLED says peaceful — false positives?
acled_merged %>%
  filter(protestor_violence == 1, sub_event_type == "Peaceful protest") %>%
  select(notes) %>%
  sample_n(10) %>%
  pull(notes) # ah, I think this looks right 

# =============================================================================
# 3. PROTEST GOALS
# =============================================================================

# Pattern matching to 'aims'; hierarchical (this took ages )
acled_merged <- acled_merged %>%
  mutate(
    notes_lower = tolower(notes),
    tags_lower  = tolower(tags),
    text_combined = paste(notes_lower, tags_lower, sep = " "),
    
    protest_goal = case_when(
      
      # Environmental
      str_detect(text_combined,
                 "climate|environment|emission|fossil fuel|pipeline|oil|fracking|
        |deforest|extinction rebellion|greenpeace|green new deal|
        |renewable|carbon|pollution|wildlife|biodiversity") ~ "Environmental",
      
      # Labour (distinct from broader fiscal)
      str_detect(text_combined,
                 "strike|walkout|picket|collective bargain|pay dispute|
        |nurses|teachers|doctors|rail workers|transport workers") ~ "Labour",
      
      # Fiscal / Economic
      str_detect(text_combined,
                 "austerity|pension|wage|cost of living|tax|budget|privatis|
        |fuel price|energy price|inflation|economic|unemployment|
        |workers|strike|trade union|labour dispute|yellow vest|gilets jaunes") ~ "Fiscal/Economic",
  
      
      # Anti-government / Political
      str_detect(text_combined,
                 "corruption|government resign|resign|no confidence|
        |democratic|election|voting|electoral|parliament|
        |political reform|constitutional") ~ "Anti-government/Political",
      
      # Immigration / Nationalism
      str_detect(text_combined,
                 "immigr|migrant|refugee|asylum|border|deportat|
        |nationalist|great replacement|remigration|
        |anti-islam|islamophob|foreigner") ~ "Immigration/Nationalism",
      
      # Anti-racism / Social Justice
      str_detect(text_combined,
                 "racism|racial|black lives matter|blm|police brutality|
        |george floyd|colonial|slavery|reparation|
        |discrimination|equality|civil rights") ~ "Anti-racism/Social Justice",
      
      # LGBTQ+
      str_detect(text_combined,
                 "lgbtq|lgbt|gay|lesbian|transgender|trans rights|
        |pride|homophob|same.sex|queer") ~ "LGBTQ+",
      
      # Abortion / Reproductive Rights
      str_detect(text_combined,
                 "abortion|reproductive|pro.life|pro.choice|
        |roe|planned parenthood|contraception") ~ "Reproductive Rights",
      
      # War / Foreign Policy
      str_detect(text_combined,
                 "war|nato|ukraine|russia|gaza|palestine|israel|
        |military|arms|weapon|bomb|sanctions|foreign policy|
        |solidarity with") ~ "War/Foreign Policy",
      
      # Far-right / Extremist (useful for partisan coding check)
      str_detect(text_combined,
                 "far.right|neo.nazi|fascis|alt.right|identitari|
        |pegida|casapound|vox|rassemblement national|
        |afd|proud boys|antifa counter") ~ "Far-right",
      
      # Housing
      str_detect(text_combined,
                 "housing|rent|evict|homeless|landlord|
        |affordable housing|gentrification") ~ "Housing",
      
      # Education
      str_detect(text_combined,
                 "education|university|tuition|student|school|
        |academic|curriculum") ~ "Education",
      
      TRUE ~ "Other/Unknown"
    )
  ) %>%
  select(-notes_lower, -tags_lower, -text_combined)  # clean up temp cols

# Check - 
table(acled_merged$protest_goal, useNA = "always") #33k unknown, but that seems fair

# Cross-tab against partisanship — key validity check
table(acled_merged$protest_goal, acled_merged$event_partisan_type_final)

# =============================================================================
# 4. NUTS-3 REGION (LOCATION TYPE)
# =============================================================================

# I think this might help for country-fixed effects later, but I actually ended up 
# abandoing this? 

# Get NUTS-3 boundaries
nuts3 <- gisco_get_nuts(
  year = "2021",
  epsg = "4326",
  resolution = "03",
  nuts_level = "3"
)

# Convert acled to sf object using lat/lon
acled_sf <- acled_merged %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# Spatial join
acled_nuts3 <- st_join(acled_sf, nuts3 %>% select(NUTS_ID, NUTS_NAME, CNTR_CODE))

# Join back to original dataframe
acled_merged <- acled_merged %>%
  left_join(
    acled_nuts3 %>% st_drop_geometry() %>% select(event_id_cnty, NUTS_ID, NUTS_NAME, CNTR_CODE),
    by = "event_id_cnty"
  )

# Check
table(is.na(acled_merged$NUTS_ID))

# Where are the unmatched events?
acled_merged %>%
  filter(is.na(NUTS_ID)) %>%
  count(country, sort = TRUE) 

table(acled_merged$NUTS_NAME)

# =============================================================================
# 5. PROXIMITY TO GOVERNMENT BUILDINGS
# =============================================================================

get_gov_points <- function(place) {
  # Get the file path without reading
  file_path <- oe_get(
    place,
    download_only = TRUE,
    quiet = FALSE
  )
  
  # Read directly via sf with SQL filter - no vectortranslate needed
  sf::st_read(
    file_path,
    layer = "points",
    query = paste("SELECT osm_id, name, amenity, building, office, geometry",
                  "FROM points WHERE",
                  "amenity IN ('townhall', 'courthouse', 'parliament', 'police', 'fire_station')",
                  "OR office IN ('government', 'parliament', 'administrative')",
                  "OR building IN ('government', 'public')",
                  "OR lower(name) LIKE '%council%'",
                  "OR lower(name) LIKE '%mairie%'",
                  "OR lower(name) LIKE '%rathaus%'",
                  "OR lower(name) LIKE '%ayuntamiento%'",
                  "OR lower(name) LIKE '%municipio%'",
                  "OR lower(name) LIKE '%prefecture%'",
                  "OR lower(name) LIKE '%ministry%'")
  ) |>
    filter(!is.na(amenity) | !is.na(office) | !is.na(building))
}

countries <- c("United Kingdom", "France", "Germany", "Italy", "Spain")

gov_points <- lapply(countries, get_gov_points) |>
  bind_rows()

# Removed all the NAs in the name (bc how am I going to verify that)
gov_points <- gov_points %>%
  filter(!is.na(name))

# Removed non-relevant public places
to_remove <- c("animal_shelter", "archive", "arts_centre", "atm", "bank", "bar",
               "bench", "bicycle_parking", "bicycle_rental", "bicycle_repair_station",
               "biergarten", "building_yard", "bus_station", "cafe", "canteen", "car_pooling",
               "car_sharing", "charging_station", "childcare", "cinema", "clinic", "clock",
               "dentist", "doctors", "drinking_water", "driving_school",
               "events_venue", "events_venue;parking", "fast_food", "ferry_terminal",
               "fountain", "ice_cream", "internet_cafe", "kindergarten", "kitchen",
               "language_school", "letter_box", "lost_property_office", "luggage_locker",
               "marketplace", "mobile_library", "motorcycle_parking", "music_school",
               "notice_board", "office", "parcel_locker", "parking", "parking_entrance",
               "parking_space", "payment_centre", "pharmacy", "place_of_worship", "post_box",
               "prep_school", "pub", "public", "public_bookcase", "recylcing", "rehabilitation",
               "research_institute", "restaurant", "serviced_office", "shelter", "taxi",
               "theatre", "ticket_validator", "toilets", "training", "vehicle_inspection",
               "vending_machine", "veterinary", "waste basket", "youth_welfare_office")
gov_points <- gov_points %>%
  filter(!amenity %in% to_remove) # There's still so much left; going to use ML 

# Classify via BART-large_MNLI
HF_TOKEN <- "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"
Sys.getenv("HF_TOKEN" #good

classify_gov_building <- function(name, amenity, office, building) {
  text <- paste(
    na.omit(c(name, amenity, office, building)),
    collapse = ", "
  )
  
  response <- request("https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli") |>
    req_headers(Authorization = paste("Bearer", Sys.getenv("HF_TOKEN"))) |>
    req_body_json(list(
      inputs = text,
      parameters = list(
        candidate_labels = c("government building", "non-government building"),
        multi_label = FALSE
      )
    )) |>
    req_perform() |>
    resp_body_json()
  
  data.frame(
    label = response[[1]]$label,
    score = response[[1]]$score
  )
}

# Attempting to run on the first row: 
classify_gov_building(gov_points$name[1], gov_points$amenity[1], 
                      gov_points$office[1], gov_points$building[1]) # 

gov_points[1, c("name", "amenity", "office", "building")] # this looks good 

# Adjusted the classifying function bc it kept failing 
classify_gov_building_safe <- function(name, amenity, office, building) {
  tryCatch(
    classify_gov_building(name, amenity, office, building),
    error = function(e) data.frame(label = NA_character_, score = NA_real_)
  )
}

# Had to chunk it up bc it kept failing the big way 

chunk_size <- 50
n <- nrow(gov_points)
chunks <- split(1:n, ceiling(seq_along(1:n) / chunk_size))

for (i in seq_along(chunks)) {
  cat("Processing chunk", i, "of", length(chunks), "\n")
  
  chunk_results <- gov_points[chunks[[i]], ] |>
    mutate(row_id = row_number()) |>
    group_by(row_id) |>
    group_modify(~ {
      Sys.sleep(0.5)
      classify_gov_building_safe(.x$name, .x$amenity, .x$office, .x$building)
    }) |>
    ungroup()
  
  saveRDS(chunk_results, paste0("chunk_", i, ".rds"))
  cat("Saved chunk", i, "\n")
}

# Reassemble at the end
all_chunks <- lapply(seq_along(chunks), function(i) {
  chunk <- readRDS(paste0("chunk_", i, ".rds"))
  chunk$row_id <- chunks[[i]]  # overwrite the broken within-chunk row_ids with global ones
  chunk
})

results <- bind_rows(all_chunks)

results_agg <- results |>
  group_by(row_id) |>
  summarise(
    label = label[which.max(replace(score, is.na(score), -Inf))],
    score = max(score, na.rm = TRUE)
  )

gov_points_classified <- gov_points |>
  mutate(row_id = row_number()) |>
  left_join(results_agg, by = "row_id") # looks good 

govt_buildings <- gov_points_classified %>% 
  filter(label == "government building")

# Reproject government buildings to EPSG:3035
govt_buildings_m <- govt_buildings %>%
  st_transform(3035)

# Distance to nearest government building (memory-safe)
nearest_gov_idx <- st_nearest_feature(acled_sf_m, govt_buildings_m)
acled_sf_m$dist_govt_building_m <- st_distance(
  acled_sf_m,
  govt_buildings_m[nearest_gov_idx, ],
  by_element = TRUE
) |> as.numeric()

# Binary indicator and proximity categories
acled_sf_m <- acled_sf_m %>%
  mutate(
    near_govt_building = as.integer(dist_govt_building_m <= 100),
    govt_proximity_cat = case_when(
      dist_govt_building_m <= 50  ~ "On/adjacent (<50m)",
      dist_govt_building_m <= 100 ~ "Very close (50-100m)",
      dist_govt_building_m <= 250 ~ "Close (100-250m)",
      dist_govt_building_m <= 500 ~ "Nearby (250-500m)",
      TRUE                        ~ "Distant (500m+)"
    )
  )

# Join back to acled_merged
acled_merged <- acled_merged %>%
  left_join(
    acled_sf_m %>%
      st_drop_geometry() %>%
      select(event_id_cnty, dist_govt_building_m, near_govt_building, govt_proximity_cat),
    by = "event_id_cnty"
  )

# Check
table(acled_merged$near_govt_building)
table(acled_merged$govt_proximity_cat)
summary(acled_merged$dist_govt_building_m)

# =============================================================================
# 6. PROXIMITY TO MAJOR ROADS
# =============================================================================

# Function to extract major roads per country
get_major_roads <- function(place) {
  file_path <- oe_get(
    place,
    download_only = TRUE,
    quiet = FALSE
  )
  
  # Check available layers first — UK gpkg sometimes differs
  layers <- sf::st_layers(file_path)$name
  target_layer <- if ("lines" %in% layers) "lines" else layers[1]
  
  sf::st_read(
    file_path,
    layer = target_layer,
    query = sprintf(
      "SELECT osm_id, name, highway, geometry
       FROM %s
       WHERE highway IN (
         'motorway', 'motorway_link',
         'trunk', 'trunk_link',
         'primary', 'primary_link',
         'secondary', 'secondary_link'
       )", target_layer
    )
  )
}

major_roads <- lapply(countries, get_major_roads) |>
  bind_rows() #jeezo 

# Reproject to a metric CRS for accurate distance calculation
# EPSG:3035 is standard for Europe
major_roads_m <- st_transform(major_roads, 3035)

acled_sf_m <- acled_merged %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(3035)

# Distance to nearest major road (in metres)
# This can be slow — consider chunking if memory is tight
nearest_idx <- st_nearest_feature(acled_sf_m, major_roads_m)
acled_sf_m$dist_major_road_m <- st_distance(
  acled_sf_m,
  major_roads_m[nearest_idx, ],
  by_element = TRUE
) |> as.numeric()

# Binary indicator: within 100m of a major road (adjust threshold as needed)
acled_sf_m <- acled_sf_m %>%
  mutate(
    near_major_road = as.integer(dist_major_road_m <= 100),
    road_proximity_cat = case_when(
      dist_major_road_m <= 50   ~ "On/adjacent (<50m)",
      dist_major_road_m <= 100  ~ "Very close (50-100m)",
      dist_major_road_m <= 250  ~ "Close (100-250m)",
      dist_major_road_m <= 500  ~ "Nearby (250-500m)",
      TRUE                      ~ "Distant (500m+)"
    )
  )

# Join back to acled_merged
acled_merged <- acled_merged %>%
  left_join(
    acled_sf_m %>%
      st_drop_geometry() %>%
      select(event_id_cnty, dist_major_road_m, near_major_road, road_proximity_cat),
    by = "event_id_cnty"
  )

# Check
table(acled_merged$near_major_road)
table(acled_merged$road_proximity_cat)

# Checking the UK bc it was giving me a tough time: 
acled_merged %>%
  filter(country == "United Kingdom") %>%
  summarise(
    n_total = n(),
    n_na_road = sum(is.na(dist_major_the road_m)),
    n_na_pct = mean(is.na(dist_major_road_m)) * 100
  ) # excellent, no missing vals 

# =============================================================================
# 7. DAY OF WEEK / WEEKEND
# =============================================================================

acled_merged <- acled_merged %>%
  mutate(
    event_date  = as.Date(event_date),
    day_of_week = weekdays(event_date),
    is_weekend  = as.integer(day_of_week %in% c("Saturday", "Sunday"))
  )

# =============================================================================
# 8. YEAR TREND
# =============================================================================

acled_merged <- acled_merged %>%
  mutate(year = lubridate::year(event_date))

# =============================================================================
# 9. URBAN/RURAL TYPOLOGY
# =============================================================================

# giscoR has NUTS with urban-rural typology attached — more complete than the
# 2013 CSV which was missing ~30% of the dataset

nuts3_2021 <- gisco_get_nuts(
  year = "2021",
  epsg = "4326", 
  resolution = "03",
  nuts_level = "3"
)

# Check if the typology column is there
names(nuts3_2021)
head(nuts3_2021[, c("NUTS_ID", "URBN_TYPE")])

acled_merged <- acled_merged %>%
  left_join(
    nuts3_2021 %>% 
      st_drop_geometry() %>%
      select(NUTS_ID, URBN_TYPE),
    by = "NUTS_ID"
  ) %>%
  mutate(
    urb_rur = factor(URBN_TYPE,
                     levels = c(1, 2, 3),
                     labels = c("Predominantly urban", "Intermediate", "Predominantly rural")
    ),
    urb_rur = relevel(urb_rur, ref = "Predominantly urban")
  )

table(acled_merged$urb_rur, useNA = "always") # so much better; okay 

# =============================================================================
# 10. Distance to Police Station (Exclusion Criteria)
# =============================================================================

# =============================================================================
# EXCLUSION RESTRICTION: DISTANCE TO NEAREST POLICE STATION
# =============================================================================

# Police stations are already in gov_points (amenity = 'police')
# Filter them out as a standalone object rather than re-downloading

# Safer: pull police stations from pre-classification data
# amenity = 'police' is unambiguous, no need for BART verification
police_stations <- gov_points %>%
  filter(amenity == "police") %>%
  st_transform(3035)

nearest_police_idx <- st_nearest_feature(acled_sf_m, police_stations)

acled_sf_m$dist_police_station_m <- st_distance(
  acled_sf_m,
  police_stations[nearest_police_idx, ],
  by_element = TRUE
) |> as.numeric()

acled_sf_m <- acled_sf_m %>%
  mutate(
    log_dist_police_station = log1p(dist_police_station_m),
    near_police_station     = as.integer(dist_police_station_m <= 500),
    police_proximity_cat    = case_when(
      dist_police_station_m <= 250  ~ "Very close (<250m)",
      dist_police_station_m <= 500  ~ "Close (250-500m)",
      dist_police_station_m <= 1000 ~ "Nearby (500m-1km)",
      dist_police_station_m <= 2000 ~ "Moderate (1-2km)",
      TRUE                          ~ "Distant (2km+)"
    )
  )

acled_merged <- acled_merged %>%
  left_join(
    acled_sf_m %>%
      st_drop_geometry() %>%
      select(event_id_cnty, dist_police_station_m,
             log_dist_police_station, near_police_station,
             police_proximity_cat),
    by = "event_id_cnty"
  )

summary(acled_merged$dist_police_station_m)
table(acled_merged$police_proximity_cat, useNA = "always")

# Sense check
acled_merged %>%
  group_by(urb_rur) %>%
  summarise(
    median_dist_km = median(dist_police_station_m, na.rm = TRUE) / 1000,
    mean_dist_km   = mean(dist_police_station_m, na.rm = TRUE) / 1000
  )

# Check for extreme outliers
acled_merged %>%
  filter(urb_rur == "Predominantly urban") %>%
  summarise(
    p95 = quantile(dist_police_station_m, 0.95, na.rm = TRUE) / 1000,
    p99 = quantile(dist_police_station_m, 0.99, na.rm = TRUE) / 1000,
    max = max(dist_police_station_m, na.rm = TRUE) / 1000
  )

# Find the worst offenders
acled_merged %>%
  filter(urb_rur == "Predominantly urban",
         dist_police_station_m > 100000) %>%  # over 100km
  select(event_id_cnty, country, location, latitude, longitude, 
         dist_police_station_m) %>%
  arrange(desc(dist_police_station_m)) %>%
  print(n = 20) # it's all the Canary Islands lol 

# How many Canary Islands events do you have total?
acled_merged %>%
  filter(longitude < -10) %>%  # west of mainland Spain
  count(location, sort = TRUE) %>%
  print(n = 20)

# I'm just going to exclude them from spatial variables: 
acled_merged <- acled_merged %>%
  mutate(
    canary_islands = as.integer(country == "Spain" & longitude < -10),
    
    # Set spatial variables to NA for Canary Islands
    dist_police_station_m   = ifelse(canary_islands == 1, NA, dist_police_station_m),
    log_dist_police_station = ifelse(canary_islands == 1, NA, log_dist_police_station),
    dist_govt_building_m    = ifelse(canary_islands == 1, NA, dist_govt_building_m),
    dist_major_road_m       = ifelse(canary_islands == 1, NA, dist_major_road_m)
  )

# =============================================================================
# SAVE
# =============================================================================

write.csv(acled_merged, 
          "~/Research/PhD/Police-Response/Current/Police_Response/data/combined/acled_merged_controls.csv",
          row.names = FALSE)

cat("\n=== Control variable construction complete ===\n")
cat("Dataset saved to acled_merged_controls.csv\n")


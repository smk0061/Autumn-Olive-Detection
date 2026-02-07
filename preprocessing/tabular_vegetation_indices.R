# tabular_vegetation_indices.R
# calculates 15 vegetation indices from tabular point-sampled spectral data.
# reads csvs with raw band values (blue, green, red, rededge, nir) and appends
# index columns. used as preprocessing for random forest classification.
#
# inputs: csvs with columns UniqueID, ClassType, ClassCode, Blue, Green, Red, RedEdge, NIR
# outputs: csvs with 15 additional vegetation index columns

library(dplyr)

# directories
input_folder  <- ""  # folder with raw spectral csvs
output_folder <- ""  # output folder for csvs with vegetation indices

dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

# helper function
safe_divide <- function(numerator, denominator) {
  ifelse(denominator == 0, 0, numerator / denominator)
}


# main processing
csv_files <- list.files(path = input_folder, pattern = "\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("no csv files found in: ", input_folder)
}

cat("found", length(csv_files), "csv files to process\n")

for (file in csv_files) {
  
  cat("processing:", basename(file), "\n")
  
  df <- read.csv(file, stringsAsFactors = FALSE)
  cat("  loaded", nrow(df), "records\n")
  
  df_clean <- df %>%
    select(UniqueID, ClassType, ClassCode, Blue, Green, Red, RedEdge, NIR)
  
  df_new <- df_clean %>%
    mutate(
      # normalized difference vegetation index
      NDVI = safe_divide(NIR - Red, NIR + Red),
      
      # normalized difference red-edge index
      NDRE = safe_divide(NIR - RedEdge, NIR + RedEdge),
      
      # green normalized difference vegetation index
      GNDVI = safe_divide(NIR - Green, NIR + Green),
      
      # blue normalized difference vegetation index
      BNDVI = safe_divide(NIR - Blue, NIR + Blue),
      
      # leaf chlorophyll index
      LCI = safe_divide(NIR - RedEdge, NIR + Red),
      
      # green chlorophyll index
      GCI = safe_divide(NIR, Green) - 1,
      
      # red edge chlorophyll index
      RECI = safe_divide(NIR, RedEdge) - 1,
      
      # simple ratio index
      SRI = safe_divide(NIR, Red),
      
      # green-red normalized difference vegetation index
      GRNDVI = safe_divide(NIR - (Green + Red), NIR + (Green + Red)),
      
      # optimized soil adjusted vegetation index
      OSAVI = safe_divide(NIR - Red, NIR + Red + 0.16),
      
      # enhanced vegetation index 2
      EVI2 = safe_divide(2.5 * (NIR - Red), NIR + 2.4 * Red + 1),
      
      # red edge green index
      ReGI = safe_divide(RedEdge, Green),
      
      # green-red vegetation index
      GRVI = safe_divide(Green - Red, Green + Red),
      
      # chlorophyll vegetation index
      CVI = safe_divide(NIR * Red, Green^2),
      
      # green-blue vegetation index
      GBVI = safe_divide(Green - Blue, Green + Blue)
    )
  
  output_file <- file.path(output_folder, basename(file))
  write.csv(df_new, output_file, row.names = FALSE)
  cat("  saved", nrow(df_new), "records to", basename(output_file), "\n")
}

cat("\nprocessed", length(csv_files), "files\n")
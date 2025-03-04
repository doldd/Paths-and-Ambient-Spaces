# Functionality to binarize regression datasets

# Load libraries
library(readr)
getwd()
dataset <- "concrete"
data_path <- "data/"
dataset_path <- paste0(data_path, dataset, ".data")
df <- readr::read_delim(dataset_path, delim = " ", col_names = FALSE)
target_median <- median(unlist(df[, ncol(df)]))
df[, ncol(df)] <- as.integer(df[, ncol(df)] > target_median)
write_delim(
  df,
  paste0(data_path, dataset, "_binary.data"),
  delim = " ", col_names = FALSE
)

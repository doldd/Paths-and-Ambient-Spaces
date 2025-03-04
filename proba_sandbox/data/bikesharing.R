library(data.table)

# Drop recording date; last column is target
bikesharing_in = fread("bikesharing.csv", drop = 2)
fwrite(bikesharing_in, "bikesharing.csv", col.names = FALSE)

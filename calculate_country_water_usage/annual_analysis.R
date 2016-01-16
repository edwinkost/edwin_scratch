
require(ggplot2)
require(foreign)

# output folder
output_folder = "/projects/0/dfguu/users/edwin/05min_runs_rerun_for_WRI_version_27_april_2015/1959_to_2010/global/analysis/country_water_use_for_WRI/table_summary/"
output_folder = "//scratch-shared/edwin/table_summary_country_water_use/"
dir.create(output_folder)

# table folder
table_folder  = "/projects/0/dfguu/users/edwin/05min_runs_rerun_for_WRI_version_27_april_2015/1959_to_2010/global/analysis/country_water_use_for_WRI/table/"

# years used
starting_year = 1960 
year_used = seq(starting_year, 2010, 1)

# get the variable names
first_year_table = read.table(paste(table_folder, "/summary_", as.character(starting_year), "-12-31.txt", sep = ""), header = T)
# - variable names starting from the column 4
variable_names = names(first_year_table)[4:length(first_year_table)]

# get the class from the shape file
shp_dbf_table = read.dbf("/projects/0/dfguu/users/edwin/data/country_shp_from_tianyi/World_Polys_High.dbf")
FID = as.character(seq(0, length(shp_dbf_table$WB_NAMES) - 1, 1))
shp_dbf_table = cbind(FID, shp_dbf_table)
shp_dbf_table = shp_dbf_table[,!names(shp_dbf_table)=="WB_RULES"]

# arranging tables 
for (i_variable in 1:length(variable_names)){

# making an initial table 
selected_table = shp_dbf_table
# add a column contains the variable name (only to ease the reading)
selected_table = cbind(variable_names[i_variable], selected_table)
names(selected_table)[1] <- as.character(variable_names[i_variable])

for (i_year in 1:length(year_used)){

# open file
complete_table = read.table(paste(table_folder, "/summary_", as.character(year_used[i_year]), "-12-31.txt", sep = ""), header = T)

# read values for this variable and
cropped_table = data.frame(as.character(complete_table$class_id),  
                           complete_table[, which(names(complete_table) == variable_names[i_variable])])
names(cropped_table)[1] <- "FID"                            
names(cropped_table)[2] <- as.character(year_used[i_year])

# merge to the selected table
selected_table = merge(selected_table, cropped_table, by = "FID", all.x = TRUE)

} # end for loop for i_year


# calculate some statistics values for the entire period
# - starting and end columns
sta_col = which(names(selected_table) == as.character(starting_year))
end_col = which(names(selected_table) == as.character(year_used[length(year_used)]))
# - average and standard deviation values
average = apply(selected_table[,sta_col:end_col], 1, mean)
std_dev = apply(selected_table[,sta_col:end_col], 1, sd)
# - auto correlations (correlation to the years)
x = NA; x = year_used
y = NA; y = selected_table[,sta_col:end_col]; y[is.na(y)] <- 0
cor_pearson  = apply(y, 1, function(y) cor(x, y, method = "pearson" ))
cor_spearman = apply(y, 1, function(y) cor(x, y, method = "spearman"))
cor_kendall  = apply(y, 1, function(y) cor(x, y, method = "kendall" ))
cor_pearson__p_value =  apply(y, 1, function(y) cor.test(x, as.numeric(y), method = "pearson" )$p.value)
cor_spearman_p_value =  apply(y, 1, function(y) cor.test(x, as.numeric(y), method = "spearman")$p.value)
cor_kendall__p_value =  apply(y, 1, function(y) cor.test(x, as.numeric(y), method = "kendall" )$p.value)
# - regression analysis (to the years)
lm_slope          = apply(y, 1, function(y) lm(y~x)$coefficients[2])
lm_r_squared      = apply(y, 1, function(y) summary( lm(y~x) )$r.squared)
lm_adj_r_squared  = apply(y, 1, function(y) summary( lm(y~x) )$r.squared)
lm_slope[is.na(average)] = NA


# calculate some statistics values for the period 1981-1990
# - starting and end columns
sta_col = which(names(selected_table) == as.character(1981))
end_col = which(names(selected_table) == as.character(1990))
# - average and standard deviation values
average_1981_to_1990 = apply(selected_table[,sta_col:end_col], 1, mean)
std_dev_1981_to_1990 = apply(selected_table[,sta_col:end_col], 1, sd)

# calculate some statistics values for the period 1991-2000
# - starting and end columns
sta_col = which(names(selected_table) == as.character(1991))
end_col = which(names(selected_table) == as.character(2000))
# - average and standard deviation values
average_1991_to_2000 = apply(selected_table[,sta_col:end_col], 1, mean)
std_dev_1991_to_2000 = apply(selected_table[,sta_col:end_col], 1, sd)

# calculate some statistics values for the period 2001-2010
# - starting and end columns
sta_col = which(names(selected_table) == as.character(2001))
end_col = which(names(selected_table) == as.character(2010))
# - average and standard deviation values
average_2001_to_2010 = apply(selected_table[,sta_col:end_col], 1, mean)
std_dev_2001_to_2010 = apply(selected_table[,sta_col:end_col], 1, sd)

# merge the aforementioned variables to the final table
final_table = cbind(selected_table, 
                    average, std_dev, cor_pearson, cor_spearman, cor_kendall, cor_pearson__p_value, cor_spearman_p_value, cor_kendall__p_value, 
                    lm_slope, lm_r_squared, lm_adj_r_squared, 
                    average_1981_to_1990, std_dev_1981_to_1990, 
                    average_1991_to_2000, std_dev_1991_to_2000, 
                    average_2001_to_2010, std_dev_2001_to_2010)
# sort table 
final_table[order(as.numeric(as.character(final_table$FID))), ]

# write the final table to a txt file
output_file_name = paste(output_folder, "/", as.character(variable_names[i_variable]), ".txt", sep = "")
print(output_file_name)
write.table(final_table, output_file_name, sep = ";", row.names = FALSE)

} # end for loop for i_variable  

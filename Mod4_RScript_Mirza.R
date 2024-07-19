# Muhammad Umer Mirza, 05/05/2024, ALY 6040
# Module 4 Technique Practice
# Data Mining Techniques

# Clean environment before running script. Code snippet from ALY 6000 class.
cat("\014")  
rm(list = ls())
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)  
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE)
options(scipen = 100) 

# Load packages
library(pacman)
p_load(tidyverse, janitor, lubridate, ggthemes, ggplot2, ggeasy, knitr, kableExtra, psych,
       corrplot, gclus, RColorBrewer, dplyr, caret, pROC, e1071)

# Load data
psd <- read.csv("Pumpkin_Seeds_Dataset.csv")

# Initial data inspection
cat("Dimensions of data:", dim(psd), "\n")
colnames(psd)
head(psd, n = 10)
str(psd)

# Count unique values for each variable
sapply(psd, function(x) length(unique(x)))
# Count missing values for each variable
sapply(psd, function(x) sum(is.na(x)))

# Check for duplicate entries
cat("Number of duplicate rows:", sum(duplicated(psd)), "\n")

# Data Cleaning
# Clean column names with janitor
psd <- psd %>% janitor::clean_names()

# Convert the class column as factor
psd$class <- factor(psd$class)

# Check structure to confirm data cleaning changes
str(psd)

# EDA

# Summary statistics for numerical variables
# List of numerical variables in the psd dataset
numerical_vars <- c("area", "perimeter", "major_axis_length", "minor_axis_length", 
                    "convex_area", "equiv_diameter", "eccentricity", "solidity", 
                    "extent", "roundness", "aspect_ration", "compactness")

# Function to calculate and round summary statistics
calculate_stats <- function(data, var) {
  stats <- c(
    Min = round(min(data[[var]], na.rm = TRUE), 2),
    Max = round(max(data[[var]], na.rm = TRUE), 2),
    Mean = round(mean(data[[var]], na.rm = TRUE), 2),
    SD = round(sd(data[[var]], na.rm = TRUE), 2),
    Median = round(median(data[[var]], na.rm = TRUE), 2),
    IQR = round(IQR(data[[var]], na.rm = TRUE), 2)
  )
  return(stats)
}

# Apply the function to each numerical variable and store results
stats_list <- lapply(numerical_vars, calculate_stats, data = psd)

summary_stats <- tibble(
  Variable = numerical_vars,
  Min = sapply(stats_list, "[", "Min"),
  Max = sapply(stats_list, "[", "Max"),
  Mean = sapply(stats_list, "[", "Mean"),
  SD = sapply(stats_list, "[", "SD"),
  Median = sapply(stats_list, "[", "Median"),
  IQR = sapply(stats_list, "[", "IQR")
)

# Print and create a table using kable
print(summary_stats)
kable(summary_stats, caption = "Summary Statistics for Numerical Variables in Pumpkin Seed Dataset", format = "html") %>%
  kable_classic(full_width = F, html_font = "Cambria")  %>%
  kable_styling(bootstrap_options = "striped", full_width = F)


# Bar chart for the categorical variable class

# Reorder class variable based on count
psd$class <- factor(psd$class, 
                               levels = names(sort(table(psd$class), decreasing = FALSE)))
ggplot(psd, aes(x = class)) +
  geom_bar(fill = "steelblue") +
  labs(x = "Class", y = "Frequency", title = "Frequency of Each Class in Pumpkin Seed Dataset",
       caption = "Data source: https://doi.org/10.1007/s10722-021-01226-0") +
  coord_flip() +  # Flips the axes to put class on the y-axis
  theme_bw() +  # Using black and white theme
  theme(axis.title = element_text(color = "black", face = "bold"),
        axis.text = element_text(color = "black"),
        axis.ticks = element_blank(),  # Removing axis ticks
        plot.title = element_text(face = "bold", hjust = 0.5, color = "black", size = 16),  # Centered title
        panel.grid = element_blank(),  # Removing grid lines
        panel.background = element_rect(fill = "white", colour = NA))  # White background without border

# Correlation matrix 
correlation_matrix <- cor(psd[, sapply(psd, is.numeric)], use = "complete.obs")
correlation_matrix <- round(correlation_matrix, 3)

# Visualize the correlation matrix using corrplot with numbers
corrplot(correlation_matrix, method = "number")

# Visualize the correlation matrix using corrplot with colors and correlation numbers
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(correlation_matrix, method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", tl.col = "black", tl.srt = 30, 
         number.cex = 0.8, diag = FALSE)
title("Correlation Matrix", line = 2)

# Data Mining
# Support Vector Machine (SVM)
# Load the necessary library
#library(e1071)

# Train SVM model using all features
m1 <- svm(class ~ ., data = psd)
summary(m1)  # Display the summary of the trained model

# Generate confusion matrix
table1 <- table(psd$class, predict(m1))
print(table1)  # Print the confusion matrix

# Calculate the classification rate
classification_rate <- sum(diag(table1)) / sum(table1)
print(classification_rate)  # Print the classification rate

# Graphical output to visualize the decision boundary for two features
plot(m1, psd, area ~ perimeter)

plot(m1, psd, area ~ roundness)

plot(m1, psd, area ~ compactness)

# Function to tune gamma and evaluate performance across different kernels
tune.gamma <- function(scale = 0.5) {
  Kernels <- c("linear", "polynomial", "radial", "sigmoid")
  gdv <- 1/12  # Default gamma value
  gDefault <- numeric(length(Kernels))  # Store default gamma results
  gLow <- numeric(length(Kernels))      # Store lower gamma results
  gHigh <- numeric(length(Kernels))     # Store higher gamma results
  
  # Loop through each kernel type and adjust gamma
  for (i in seq_along(Kernels)) {
    # Test default gamma
    tmp <- svm(class ~ ., data = psd, kernel = Kernels[i], gamma = gdv)
    tbl <- table(psd$class, predict(tmp))
    gDefault[i] <- sum(diag(tbl)) / sum(tbl)
    
    # Test gamma scaled down
    tmp <- svm(class ~ ., data = psd, kernel = Kernels[i], gamma = gdv - scale * gdv)
    tbl <- table(psd$class, predict(tmp))
    gLow[i] <- sum(diag(tbl)) / sum(tbl)
    
    # Test gamma scaled up
    tmp <- svm(class ~ ., data = psd, kernel = Kernels[i], gamma = gdv + scale * gdv)
    tbl <- table(psd$class, predict(tmp))
    gHigh[i] <- sum(diag(tbl)) / sum(tbl)
  }
  
  # Return a data frame of results for each kernel
  return(data.frame(Kernels, gLow, gDefault, gHigh))
}

# Execute gamma tuning
gamma_tuning_results <- tune.gamma(0.5)
print(gamma_tuning_results)


# End of analysis

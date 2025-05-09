# load data
#install.packages('ISLR','lubridate','janitor','tidyr','pROC','caret','reshape2')
library(ISLR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(janitor)
library(reshape2)
library(ggplot2)
library(pROC)
library(caret)
library(kableExtra)

# Function to create a glimpse table
create_glimpse_table <- function(df) {
  tibble(
    Column_Name = names(df),
    Data_Type = sapply(df, class),
    Example_Value = sapply(df, function(x) if (length(x) > 0) x[1] else NA)
  )
}

#Glimpse of Data
raw_data_glimpse<-create_glimpse_table(College)
# Summary of Numeric columns
summary(College)
names(College)

# Check for missing values
sum(is.na(College)) # no missing values
College<-janitor::clean_names(College)
names(College)

############################################# EDA ############################################# 
# Count of Private colleges (Yes/No)
College %>%
  count(private) %>%
  ggplot(aes(x = private, y = n, fill = private)) +
  geom_bar(stat = "identity") +
  labs(title = "Count of Private Colleges", x = "Private", y = "Count") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_text(size = 10, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 10)
  )

# Histogram of Outstate tuition
ggplot(College, aes(x = outstate)) +
  geom_histogram(binwidth = 1000, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Outstate Tuition", x = "Outstate Tuition", y = "Count") +
  theme_minimal()


# Compute correlations for numeric variables
numeric_vars <- select(College, where(is.numeric))
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualize correlations
cor_df <- melt(cor_matrix)
ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#############################################  EDA ############################################# 

#################################### Select Predictors : Step-Wise Regression #######################################
# Convert 'Private' to a binary variable (1 for 'Yes', 0 for 'No')
College$private <- ifelse(College$private == "Yes", 1, 0)

# Fit a logistic regression model with stepwise selection
model_initial <- glm(private ~ 1, data = College, family = "binomial")  # Null model
model_full <- glm(private ~ ., data = College, family = "binomial")    # Full model

# Perform stepwise selection
model_step <- step(model_initial, scope = formula(model_full), direction = "both")

# Summary of the final model
summary(model_step)
# Extract the anova table from the stepwise model, which contains the step information.
step_info <- model_step$anova

# Create a data frame to store step information for plotting
stepwise_df <- data.frame(
  Step = 1:nrow(step_info),
  AIC = step_info$AIC,
  Variable = step_info$Step
)

# Use kable to display the table in a report-friendly format
kable(stepwise_df, 
      caption = "Stepwise Regression: Steps, AIC, and Variables Added/Removed",
      col.names = c("Step", "AIC", "Variable Added/Removed"),
      format = "markdown") 

# Print the table for reporting
print(stepwise_df)
#################################### Step-Wise Regression #######################################

#################################### Fit Logistic Regression Model - Train Set ####################################
# Split the data into training and testing sets
set.seed(42)  # For reproducibility
train_indices <- sample(1:nrow(College), size = 0.7 * nrow(College))
train_set <- College[train_indices, ]
test_set <- College[-train_indices, ]

# Fit logistic regression model using glm()
logistic_model <- glm(private ~ f_undergrad + outstate + grad_rate + ph_d + expend , 
                      data = train_set, 
                      family = binomial)

# Summary of the model
summary(logistic_model)


# Predict on the train set
train_pred_prob <- predict(logistic_model, train_set, type = "response")
train_pred <- ifelse(train_pred_prob > 0.8, 1, 0)

# Confusion Matrix
conf_matrix_train <- confusionMatrix(as.factor(train_pred), as.factor(train_set$private))

# Print the confusion matrix
print(conf_matrix_train)


# Extract metrics
metrics_train <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall (Sensitivity)", "Specificity"),
  Value = c(
    conf_matrix_train$overall["Accuracy"],
    conf_matrix_train$byClass["Precision"], # Precision
    conf_matrix_train$byClass["Recall"],   # Recall
    conf_matrix_train$byClass["Specificity"]    # Specificity
  )
)

# Print as a kable
kable(metrics_train, col.names = c("Metric", "Value"), digits = 4, caption = "Confusion Matrix Metrics (Train Set)")
#################################### Fit Logistic Regression Model - Train Set####################################

#################################### Model Confusion Matrix & ROC Curve - Test Set ####################################
# Predict on the test set
test_set$predicted_prob <- predict(logistic_model, newdata = test_set, type = "response")
test_set$predicted_class <- ifelse(test_set$predicted_prob > 0.8, 1, 0)

# Confusion Matrix for the Test Set
conf_matrix_test <- confusionMatrix(as.factor(test_set$predicted_class), as.factor(test_set$private))

# Print the confusion matrix for the test set
print(conf_matrix_test)

# Extract metrics
metrics_test <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall (Sensitivity)", "Specificity"),
  Value = c(
    conf_matrix_test$overall["Accuracy"],
    conf_matrix_test$byClass["Precision"], # Precision
    conf_matrix_test$byClass["Recall"],   # Recall
    conf_matrix_test$byClass["Specificity"]    # Specificity
  )
)

# Print as a kable
kable(metrics_test, col.names = c("Metric", "Value"), digits = 4, caption = "Confusion Matrix Metrics (Test Set)")


# ROC Curve
roc_curve <- roc(test_set$private, test_set$predicted_prob)

# Plot the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve")
#abline(a = 0, b = 1, lty = 2, col = "gray")

# Calculate AUC
auc_value <- auc(roc_curve)
cat("\nAUC:", auc_value, "\n")

#################################### Model Confusion Matrix & ROC Curve - Test Set  ####################################
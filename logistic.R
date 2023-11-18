# Load necessary libraries
library(ggplot2)
library(caret)

# import data set
pima_data <- read.csv('pima-indians-diabetes.csv', header = FALSE)
colnames(pima_data) <- c("num_of_pregnant", "plasma_glucose_concentration", "blood_pressure",
                                          "skinfold_thickness", "serum_insulin", "bmi",
                                          "diabetes_predigree", "age",  "diabetes")
#head(pima_data, 5)

# Scatterplot
ggplot(pima_data, aes(x=plasma_glucose_concentration, y=age, color=as.factor(diabetes))) +
  geom_point() +
  labs(color="diabetes", title='Scatter Plot of Plasma Glucose Concentration on Age')


# Logistic Regression
logsitic_model <- glm(diabetes ~ plasma_glucose_concentration + age, data=pima_data, family=binomial)
summary(logsitic_model)

# Predict probabilities
probabilities <- predict(logsitic_model, type="response")
print(paste0("Probability(Diabetes=1) = 1 / (1 + exp(-(",coef(logsitic_model)[1]," + ", coef(logsitic_model)[2], "*x1 + ", coef(logsitic_model)[3], "*x2)))"))

# Classify observations
predictions <- ifelse(probabilities >= 0.5, 1, 0)

# Compute misclassification error
mis_error <- mean(predictions != pima_data$diabetes)
cat("Misclassification Error:", mis_error, "\n")

# Scatter plot with predicted values
ggplot(data.frame(pima_data, predictions), aes(x=plasma_glucose_concentration, y=age, color=as.factor(predictions))) +
  geom_point() +
  labs(color="predictions",title='Scatter Plot with Predicted Diabetes Values')



# Decision boundary equation
# Decision boundary is where the logistic function equals 0.5
# 0 = intercept + coef1*x1 + coef2*x2
# x2 = -(intercept + coef1*x1) / coef2

#decision_boundary <- -(coef(logsitic_model)[1] + coef(logsitic_model)[2]*pima_data$age) / coef(logsitic_model)[3]
decision_boundary_x1 <- pima_data$plasma_glucose_concentration
decision_boundary_x2 <- -(coef(logsitic_model)[1] + coef(logsitic_model)[2]*decision_boundary_x1) / coef(logsitic_model)[3]

# Add decision boundary to scatter plot
ggplot(data.frame(pima_data, predictions), aes(x=plasma_glucose_concentration, y=age, color=as.factor(predictions))) +
  geom_point() +
  geom_line(aes(x=decision_boundary_x1, y=decision_boundary_x2), color='red', linetype='dashed', linewidth=1) + xlim(0, 200) + ylim(20, 80) +
  labs(color="diabetes", title='Scatter Plot of Plasma Glucose Concentration on Age with Decision Boundary')

# Predictions with different thresholds
thresholds <- c(0.2, 0.8)

for ( threshold in thresholds) {
  pred_threshold <- ifelse(predict(logsitic_model, newdata=pima_data, type='response') >= threshold, 1, 0)

  # Scatter plot with predicted values and threshold
  scatter_plot <- ggplot(data.frame(pima_data, pred_threshold), aes(x=plasma_glucose_concentration, y=age, color=as.factor(pred_threshold))) +
    geom_point() +
    labs(color="pred_threshold", title=paste('Scatter Plot with Predicted Diabetes Values (Threshold =', threshold, ')'))
  print(scatter_plot)
}


# Create new features
pima_data$z1 <- pima_data$plasma_glucose_concentration^4
pima_data$z2 <- pima_data$plasma_glucose_concentration^3 * pima_data$age
pima_data$z3 <- pima_data$plasma_glucose_concentration^2 * pima_data$age^2
pima_data$z4 <- pima_data$plasma_glucose_concentration * pima_data$age^3
pima_data$z5 <- pima_data$age^4

# Logistic Regression with new features
model_expanded <- glm(diabetes ~ plasma_glucose_concentration + age + z1 + z2 + z3 + z4 + z5, data=pima_data, family=binomial)


# Predict probabilities and classify observations
probabilities_expanded <- predict(model_expanded, type="response")
predictions_expanded <- ifelse(probabilities_expanded >= 0.5, 1, 0)


# Scatter plot for the model with expanded features
ggplot(data.frame(pima_data, predictions_expanded), aes(x=plasma_glucose_concentration, y=age, color=as.factor(predictions_expanded))) +
  geom_point() +
  labs(color="predictions_expanded",title='Scatter Plot with Predicted Diabetes Values (Expanded Features)')


# Compute misclassification error
mis_error_expanded <- mean(predictions_expanded != pima_data$diabetes)
cat("Misclassification Error (Expanded Features):", mis_error_expanded, "\n")

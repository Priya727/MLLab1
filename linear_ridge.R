# import data set
parkinson_data <- read.csv('parkinsons.csv', header = TRUE)
head(parkinson_data, 5)

#parkinson_data <- scale(parkinson_data)

# Split the data into training and test sets
parkinson_data <- subset(parkinson_data, select = -c(1:4,6))
n=dim(parkinson_data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.6))
train_data=parkinson_data[id,]
test_data=parkinson_data[-id,]

# Scale the data
#scaled_train_data <- as.data.frame(cbind(scale(subset(train_data, select = -c(motor_UPDRS))), motor_UPDRS=train_data$motor_UPDRS))
#scaled_test_data <- as.data.frame(cbind(scale(subset(test_data, select = -c(motor_UPDRS))), motor_UPDRS=test_data$motor_UPDRS))

scaled_train_data <- as.data.frame(scale(train_data))
scaled_test_data <- as.data.frame(scale(test_data))



# linear regression model
lm_model <- lm(motor_UPDRS ~ ., data = scaled_train_data)

# Training and test MSE
train_pred <- predict(lm_model, newdata = scaled_train_data)
train_mse <- mean((scaled_train_data$motor_UPDRS - train_pred)^2)
cat("The training data's MSE is: ", train_mse)

test_pred <- predict(lm_model, newdata = scaled_test_data)
test_mse <- mean((scaled_test_data$motor_UPDRS - test_pred)^2)
cat("The training data's MSE is: ", test_mse)

# Significant variables
summary(lm_model)

# Log-likelihood function
log_likelihood_fun <- function(theta, sigma, y, X) {
  n <- length(y)
  log_likelihood <- -n/2 * log(2*pi*sigma^2) - 1/(2*sigma^2) * sum((y - X%*%theta)^2)
  return(log_likelihood)
}

# Ridge log-likelihood function
ridge_log_Likelihood_fun <- function(theta, sigma, lambda, y, X) {
  log_like <- log_likelihood_fun(theta, sigma, y, X)
  ridge_penalty <- lambda * sum(theta^2)
  return(-log_like + ridge_penalty)
}

# Ridge log-likelihood optimization function
ridge_log_likelihood_opt <- function(lambda, y, X) {
  start <- c(rep(0, ncol(X)), sd(y)) # combine theta and sigma into one numeric vector
  theta_length <- ncol(X)

  # Define a function for the optim() call
  fn_to_optim <- function(params) {
    theta <- params[1:theta_length]
    sigma <- params[theta_length + 1]
    return(-ridge_log_Likelihood_fun(theta, sigma, lambda, y, X))
  }

  # Call optim()
  opt_res <- optim(start, fn = fn_to_optim, method = "BFGS")

  # Return theta and sigma separately
  return(list(theta = opt_res$par[1:theta_length], sigma = opt_res$par[theta_length + 1]))
}


# Degrees of freedom function
df_fun <- function(lambda, X) {
  H <- solve(t(X) %*% X + lambda * diag(ncol(X))) %*% t(X) %*% X
  return(sum(diag(H)))
}

y <- scaled_train_data$motor_UPDRS
X <- as.matrix(subset(scaled_train_data, select = -c(motor_UPDRS)))
# Ridge optimization for different lambdas
lambdas <- c(1, 100, 1000)
opt_params <- lapply(lambdas, function(lambda) ridge_log_likelihood_opt(lambda, y, X))

# Predictions and MSE for different lambdas
ridge_train_preds <- lapply(1:length(opt_params), function(i) {
  theta <- opt_params[[i]]$theta
  as.matrix(subset(scaled_train_data, select = -c(motor_UPDRS))) %*% theta
})

#train_mses <- colMeans((scaled_train_data$motor_UPDRS - ridge_train_preds)^2)
ridge_train_mse <- sapply(ridge_train_preds, function(pred) mean((pred - scaled_train_data$motor_UPDRS)^2))
cat("Ridge Training MSE:", ridge_train_mse, "\n")

ridge_test_preds <- lapply(1:length(opt_params), function(i) {
  theta <- opt_params[[i]]$theta
  as.matrix(subset(scaled_test_data, select = -c(motor_UPDRS))) %*% theta
})

ridge_test_mse <- sapply(ridge_test_preds, function(pred) mean((pred - scaled_test_data$motor_UPDRS)^2))
cat("Ridge Test MSE:", ridge_test_mse, "\n")


# Degrees of freedom for different lambdas
degrees_of_freedom <- sapply(lambdas, function(lambda) df_fun(lambda, X))
cat("Degrees of Freedom:", degrees_of_freedom, "\n")



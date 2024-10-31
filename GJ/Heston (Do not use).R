# Stock tickers
ASSET_NAMES <- c('LONN.SE','SIKA.SE')

INITIAL_LEVELS <- data.frame(
  Underlying_assets = c("LONN.SE",'SIKA.SE'),
  value = c(549.60, 240.40))
CONVERSION_RATIOS <- data.frame(
  Underlying_assets = c("LONN.SE",'SIKA.SE'),
  value = c(1.8195, 4.1597))


# Information on factsheet
INITIAL_FIXING_DATE <- as.Date("2023-04-27")
PAYMENT_DATE <- as.Date("2023-05-05")
FINAL_FIXING_DATE <- as.Date("2024-07-30")
REDEMPTION_DATE <- as.Date("2024-08-05")

# If all Reference Shares close at or above their Early Redemption Levels on any Early Redemption Observation Date
EARLY_REDEMPTION_LEVEL <- 1.00  # 100% of the Initial Level
EARLY_REDEMPTION_OBSERVATION_FREQUENCY <- "quarterly"

CURRENCY <- "CHF"
DENOMINATION <- 1000  # CHF 1,000
ISSUE_PRICE_PERCENTAGE <- 1.00  # 100%

# Simulation constants
INITIAL_PROD_PRICING_DATE <- as.Date("2024-08-01")
FINAL_PROD_PRICING_DATE <- as.Date("2024-11-01")
PRICING_WINDOW <- 60
INTEREST_RATE <- 1.750 / 100
# SIMULATION_START_DATE <- next day from the date of the product price estimation

# Historical stock prices to fetch
HISTORICAL_START_DATE <- as.Date("2022-08-09")
HISTORICAL_END_DATE <- Sys.Date()

DENOMINATION <- 1000.0
BARRIER <- 0.6
COUPON_RATE <- 0.08 / 4
COUPON_PAYOUT <- COUPON_RATE * DENOMINATION

COUPON_PAYMENT_DATES <- as.Date(c(
  "2023-08-07",
  "2023-11-06",
  "2024-02-05",
  "2024-05-06",
  "2024-08-05"
))

EARLY_REDEMPTION_OBSERVATION_DATES <- as.Date(c(
  "2023-11-01",
  "2024-01-31",
  "2024-04-30"
))

EARLY_REDEMPTION_DATES <- data.frame(
  observation_date = as.Date(c("2023-11-01", "2024-01-31", "2024-04-30")),
  redemption_date = as.Date(c("2023-11-06", "2024-02-05", "2024-05-06"))
)

SIX_HOLIDAY_DATES <- as.Date(c(
  "2023-01-02",  # Berchtoldstag
  "2023-04-07",  # Good Friday
  "2023-04-10",  # Easter Monday
  "2023-05-01",  # Labour Day
  "2023-05-18",  # Ascension Day
  "2023-05-29",  # Whit Monday
  "2023-08-01",  # National Day
  "2023-12-25",  # Christmas Day
  "2023-12-26",  # St. Stephen's Day
  "2024-01-01",  # New Year's Day
  "2024-01-02",  # Berchtoldstag
  "2024-03-29",  # Good Friday
  "2024-04-01",  # Easter Monday
  "2024-05-01",  # Labour Day
  "2024-05-09",  # Ascension Day
  "2024-05-20",  # Whit Monday
  "2024-08-01",  # National Day
  "2024-12-24",  # Christmas Eve
  "2024-12-25",  # Christmas Day
  "2024-12-26",  # St. Stephen's Day
  "2024-12-31"   # New Year's Eve
))

#ESTIMATION OF PARAMETERS (MODEL CALIBRATION)
library(MASS)    # For multivariate normal distribution
library(stats)   # For optimization functions

# Heston characteristic function
heston_cf <- function(u, params, S0, r, T) {
  i <- complex(real = 0, imaginary = 1)
  kappa <- params$kappa
  theta <- params$theta
  sigma_v <- params$sigma_v
  rho <- params$rho
  v0 <- params$v0
  
  lambda <- 0  # Risk parameter, often set to zero
  d <- sqrt((rho * sigma_v * i * u - kappa - lambda)^2 + sigma_v^2 * (i * u + u^2))
  g <- (kappa + lambda - rho * sigma_v * i * u + d) / (kappa + lambda - rho * sigma_v * i * u - d)
  
  C <- r * i * u * T + (kappa * theta) / (sigma_v^2) * ((kappa + lambda - rho * sigma_v * i * u + d) * T - 2 * log((1 - g * exp(d * T)) / (1 - g)))
  D <- ((kappa + lambda - rho * sigma_v * i * u + d) / sigma_v^2) * ((1 - exp(d * T)) / (1 - g * exp(d * T)))
  
  cf_value <- exp(C + D * v0 + i * u * log(S0))
  return(cf_value)
}

# COS method for option pricing
cos_method_heston <- function(S0, K, r, T, params, N = 256, L = 10) {
  i <- complex(real = 0, imaginary = 1)
  
  c1 <- log(S0 / K) + (r - 0.5 * params$v0) * T
  c2 <- params$v0 * T  # Approximate variance over [0, T]
  a <- c1 - L * sqrt(abs(c2))
  b <- c1 + L * sqrt(abs(c2))
  
  k <- 0:(N - 1)
  u <- k * pi / (b - a)
  
  # Characteristic function values
  cf_values <- sapply(u, function(u_j) {
    cf <- heston_cf(u_j, params, S0, r, T)
    return(Re(cf * exp(-i * u_j * a)))
  })
  
  # Handle the division by zero for u = 0
  payoff_coefficients <- numeric(N)
  payoff_coefficients[1] <- K * (b - a)
  payoff_coefficients[-1] <- 2 * K * sin(u[-1] * (b - a) / 2) / u[-1]
  
  # Option price
  option_price <- exp(-r * T) * sum(payoff_coefficients * cf_values) / (b - a)
  
  return(option_price)
}

# Function to compute implied volatility
implied_volatility <- function(price, S, K, r, T) {
  black_scholes_call <- function(sigma) {
    d1 <- (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 <- d1 - sigma * sqrt(T)
    bs_price <- S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
    return(bs_price)
  }
  
  # Define the objective function for root-finding
  objective_function <- function(sigma) {
    bs_price <- black_scholes_call(sigma)
    return(bs_price - price)
  }
  
  # Use uniroot to find the implied volatility
  implied_vol <- tryCatch(
    uniroot(objective_function, lower = 1e-6, upper = 5)$root,
    error = function(e) { NA }
  )
  return(implied_vol)
}

# Function to compute model-implied volatility under Heston using COS method
heston_model_iv <- function(params_vector, S0, K, r, T_mat) {
  params <- list(
    kappa = params_vector[1],
    theta = params_vector[2],
    sigma_v = params_vector[3],
    rho = params_vector[4],
    v0 = params_vector[5]
  )
  
  # Compute the option price using the Heston model via COS method
  price <- cos_method_heston(S0, K, r, T_mat, params)
  
  # Compute the implied volatility from the model price
  model_iv <- implied_volatility(price, S0, K, r, T_mat)
  
  return(model_iv)
}

# Objective function for calibration
objective_function <- function(params_vector, market_data, S0, r, ...) {
  errors <- sapply(1:nrow(market_data), function(i) {
    K <- market_data$Strike[i]
    T_mat <- market_data$Maturity[i]
    market_iv <- market_data$Market_IV[i]
    
    model_iv <- heston_model_iv(params_vector, S0, K, r, T_mat)
    
    if (is.na(model_iv)) {
      return(1e6)  # Assign a large error if implied volatility calculation fails
    } else {
      return((model_iv - market_iv)^2)
    }
  })
  return(sum(errors))
}

# Calibration for Lonza Group AG
# Replace with actual data from Bloomberg
option_data_LONN <- data.frame(
  Strike = c(500, 550, 600),
  Maturity = c(1, 1, 1),
  Market_IV = c(0.25, 0.22, 0.20)
)

# Initial parameter guesses for Lonza
initial_params_LONN <- c(kappa = 2, theta = 0.04, sigma_v = 0.3, rho = -0.7, v0 = 0.04)

# Calibrate parameters for Lonza Group AG
calibration_result_LONN <- optim(
  par = initial_params_LONN,
  fn = objective_function,
  method = "L-BFGS-B",
  lower = c(0.01, 0.01, 0.01, -0.999, 0.01),
  upper = c(10, 1, 1, 0.999, 1),
  market_data = option_data_LONN,
  S0 = 549.60,
  r = 0.01
)

# Calibrated parameters for Lonza
params_calibrated_LONN <- calibration_result_LONN$par
print("Calibrated Parameters for Lonza Group AG:")
print(params_calibrated_LONN)

# Calibration for Sika AG
# Replace with actual data from Bloomberg
option_data_SIKA <- data.frame(
  Strike = c(220, 240, 260),
  Maturity = c(1, 1, 1),
  Market_IV = c(0.30, 0.28, 0.26)
)

# Initial parameter guesses for Sika
initial_params_SIKA <- c(kappa = 2, theta = 0.05, sigma_v = 0.35, rho = -0.6, v0 = 0.05)

# Calibrate parameters for Sika AG
calibration_result_SIKA <- optim(
  par = initial_params_SIKA,
  fn = objective_function,
  method = "L-BFGS-B",
  lower = c(0.01, 0.01, 0.01, -0.999, 0.01),
  upper = c(10, 1, 1, 0.999, 1),
  market_data = option_data_SIKA,
  S0 = 240.40,
  r = 0.01
)

# Calibrated parameters for Sika
params_calibrated_SIKA <- calibration_result_SIKA$par
print("Calibrated Parameters for Sika AG:")
print(params_calibrated_SIKA)

# Correlation coefficients
rho_assets <- 0.5    # Correlation between the two assets (estimate from historical data)
rho_v1 <- params_calibrated_LONN[4]  # Calibrated rho for Lonza
rho_v2 <- params_calibrated_SIKA[4]  # Calibrated rho for Sika

# Construct the covariance matrix for the Brownian motions
cov_matrix <- matrix(c(
  1,          rho_assets, rho_v1,       0,
  rho_assets, 1,          0,            rho_v2,
  rho_v1,     0,          1,            0,
  0,          rho_v2,     0,            1
), nrow = 4, ncol = 4)

# Ensure the covariance matrix is positive semi-definite
eigenvalues <- eigen(cov_matrix)$values
if (any(eigenvalues < 0)) {
  stop("Covariance matrix is not positive semi-definite. Please adjust correlation parameters.")
}

# Simulate the multidimensional Heston model
simulate_multidimensional_heston <- function(params1, params2, S0_1, S0_2, r, T, N, M, cov_matrix) {
  dt <- T / N
  S1 <- matrix(0, nrow = M, ncol = N + 1)
  v1 <- matrix(0, nrow = M, ncol = N + 1)
  S2 <- matrix(0, nrow = M, ncol = N + 1)
  v2 <- matrix(0, nrow = M, ncol = N + 1)
  
  S1[, 1] <- S0_1
  v1[, 1] <- params1$v0
  S2[, 1] <- S0_2
  v2[, 1] <- params2$v0
  
  # Cholesky decomposition of the covariance matrix
  L <- t(chol(cov_matrix))
  
  for (i in 1:N) {
    # Generate standard normal random variables
    Z <- matrix(rnorm(M * 4), nrow = M, ncol = 4)
    # Apply Cholesky decomposition to induce correlations
    dW <- Z %*% L * sqrt(dt)
    
    # Asset 1
    v1_prev <- v1[, i]
    v1_new <- v1_prev + params1$kappa * (params1$theta - v1_prev) * dt +
      params1$sigma_v * sqrt(pmax(v1_prev, 0)) * dW[, 3]
    v1_new <- pmax(v1_new, 0)
    v1[, i + 1] <- v1_new
    
    S1_prev <- S1[, i]
    S1_new <- S1_prev * exp((r - 0.5 * v1_prev) * dt + sqrt(pmax(v1_prev, 0)) * dW[, 1])
    S1[, i + 1] <- S1_new
    
    # Asset 2
    v2_prev <- v2[, i]
    v2_new <- v2_prev + params2$kappa * (params2$theta - v2_prev) * dt +
      params2$sigma_v * sqrt(pmax(v2_prev, 0)) * dW[, 4]
    v2_new <- pmax(v2_new, 0)
    v2[, i + 1] <- v2_new
    
    S2_prev <- S2[, i]
    S2_new <- S2_prev * exp((r - 0.5 * v2_prev) * dt + sqrt(pmax(v2_prev, 0)) * dW[, 2])
    S2[, i + 1] <- S2_new
  }
  
  return(list(S1 = S1, v1 = v1, S2 = S2, v2 = v2))
}

# Parameters for Lonza Group AG
params1 <- list(
  kappa = params_calibrated_LONN[1],
  theta = params_calibrated_LONN[2],
  sigma_v = params_calibrated_LONN[3],
  rho = params_calibrated_LONN[4],
  v0 = params_calibrated_LONN[5]
)
S0_LONN <- 549.60

# Parameters for Sika AG
params2 <- list(
  kappa = params_calibrated_SIKA[1],
  theta = params_calibrated_SIKA[2],
  sigma_v = params_calibrated_SIKA[3],
  rho = params_calibrated_SIKA[4],
  v0 = params_calibrated_SIKA[5]
)
S0_SIKA <- 240.40

# Simulate the paths
simulations <- simulate_multidimensional_heston(
  params1, params2, S0_LONN, S0_SIKA, r = 0.01, T = 1.25, N = 250, M = 10000, cov_matrix
)

# Extract the simulated asset price paths
S_LONN <- simulations$S1
S_SIKA <- simulations$S2

# Define product parameters
params_product <- list(
  Denomination = 1000,
  Coupon = 0.08 / 4 * 1000,  # Quarterly coupon payments
  Barrier_LONN = 329.76,
  Barrier_SIKA = 144.24,
  Early_Redemption_Level_LONN = S0_LONN,
  Early_Redemption_Level_SIKA = S0_SIKA,
  Conversion_Ratio_LONN = 1.8195,
  Conversion_Ratio_SIKA = 4.1597,
  S0_LONN = S0_LONN,
  S0_SIKA = S0_SIKA
)

# Compute the payoff function
compute_payoff <- function(S_LONN, S_SIKA, params) {
  M <- nrow(S_LONN)
  N <- ncol(S_LONN) - 1  # Adjust for the initial time step
  payoff <- numeric(M)
  
  # Early Redemption Dates (quarterly after 6 months)
  early_dates <- seq(0.5, 1.25, by = 0.25)  # In years
  T_total <- 1.25  # Total time in years
  dt <- T_total / N  # Time step size
  early_indices <- as.integer(early_dates / dt) + 1  # Map times to indices
  
  for (i in 1:M) {
    early_redeemed <- FALSE
    for (j in seq_along(early_indices)) {
      t_index <- early_indices[j]
      if (S_LONN[i, t_index] >= params$Early_Redemption_Level_LONN &&
          S_SIKA[i, t_index] >= params$Early_Redemption_Level_SIKA) {
        # Early Redemption
        periods <- j  # Number of coupon periods paid
        payoff[i] <- params$Denomination + params$Coupon * periods
        early_redeemed <- TRUE
        break
      }
    }
    if (!early_redeemed) {
      # Check for barrier breach
      barrier_breach <- any(S_LONN[i, ] <= params$Barrier_LONN) || any(S_SIKA[i, ] <= params$Barrier_SIKA)
      final_price_LONN <- S_LONN[i, N + 1]
      final_price_SIKA <- S_SIKA[i, N + 1]
      
      periods <- length(early_dates)  # Total periods if held to maturity
      
      if (!barrier_breach || (final_price_LONN >= params$S0_LONN && final_price_SIKA >= params$S0_SIKA)) {
        # Full redemption
        payoff[i] <- params$Denomination + params$Coupon * periods
      } else {
        # Convert to worst-performing asset
        performance_LONN <- final_price_LONN / params$S0_LONN
        performance_SIKA <- final_price_SIKA / params$S0_SIKA
        
        if (performance_LONN <= performance_SIKA) {
          # Lonza is worst-performing
          shares <- params$Denomination / params$S0_LONN
          payoff[i] <- final_price_LONN * shares
        } else {
          # Sika is worst-performing
          shares <- params$Denomination / params$S0_SIKA
          payoff[i] <- final_price_SIKA * shares
        }
        payoff[i] <- payoff[i] + params$Coupon * periods
      }
    }
  }
  return(payoff)
}

# Compute the payoff
payoffs <- compute_payoff(S_LONN, S_SIKA, params_product)

# Calculate expected payoff and present value
expected_payoff <- mean(payoffs)
present_value <- expected_payoff / exp(0.01 * 1.25)

# Output the present value
print(paste("The present value of the product is:", round(present_value, 2)))

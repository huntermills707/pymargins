# Generate survival reference data for test_correctness_survival.py
# Matches the DGP described in the Python fixture.

set.seed(42)
n <- 200

x1 <- rnorm(n, 0, 1)
x2 <- rnorm(n, 0, 1)
hazard <- exp(0.5 + 0.3 * x1 - 0.2 * x2)
time <- rexp(n, rate = hazard)
status <- as.integer(runif(n) < 0.8)

df <- data.frame(x1 = x1, x2 = x2, time = time, status = status)
write.csv(df, "/tmp/survival_data.csv", row.names = FALSE)

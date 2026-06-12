source("tools/oracle/lib.R")
library(marginaleffects)

df <- read.csv("tests/oracle/data/oracle_main.csv")
fit <- glm(y_bin ~ treat + x1 + x2, family = binomial(probit), data = df,
           control = glm.control(epsilon = 1e-12, maxit = 200))
mod <- list(formula = "y_bin ~ treat + x1 + x2",
            family = "binomial(probit)",
            fit_control = "glm.control(epsilon = 1e-12, maxit = 200)")

s <- avg_slopes(fit, variables = "x1")
write_golden("probit_ame_x1_nonrobust", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1')",
             list(coefficients = unname(coef(fit)),
                  estimate = s$estimate,
                  std_error = s$std.error,
                  conf_low = s$conf.low,
                  conf_high = s$conf.high),
             labels = list("x1"),
             tolerances = list(std_error = 0.007),
             notes = "probit nonrobust SE: R uses expected (Fisher) information, statsmodels uses observed information for non-canonical link; ~0.5% gap")

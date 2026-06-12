source("tools/oracle/lib.R")
library(marginaleffects)
library(sandwich)

df <- read.csv("tests/oracle/data/oracle_main.csv")
fit <- glm(y_count ~ treat + x1 + x2, family = poisson(), data = df,
           control = glm.control(epsilon = 1e-12, maxit = 200))
mod <- list(formula = "y_count ~ treat + x1 + x2",
            family = "poisson(log)",
            fit_control = "glm.control(epsilon = 1e-12, maxit = 200)")

# predict overall
p <- avg_predictions(fit)
write_golden("poisson_predict_overall_nonrobust", "oracle_main.csv", mod,
             "avg_predictions(fit)",
             list(coefficients = unname(coef(fit)),
                  estimate = p$estimate,
                  std_error = p$std.error,
                  conf_low = p$conf.low,
                  conf_high = p$conf.high))

# AME x1
s <- avg_slopes(fit, variables = "x1")
write_golden("poisson_ame_x1_nonrobust", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1')",
             list(coefficients = unname(coef(fit)),
                  estimate = s$estimate,
                  std_error = s$std.error,
                  conf_low = s$conf.low,
                  conf_high = s$conf.high),
             labels = list("x1"))

# AME x1 HC1
sh <- avg_slopes(fit, variables = "x1", vcov = vcovHC(fit, type = "HC1"))
write_golden("poisson_ame_x1_hc1", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', vcov = vcovHC(fit, type = 'HC1'))",
             list(coefficients = unname(coef(fit)),
                  estimate = sh$estimate,
                  std_error = sh$std.error,
                  conf_low = sh$conf.low,
                  conf_high = sh$conf.high),
             vcov = "HC1", labels = list("x1"),
             tolerances = list(std_error = 0.006),
             notes = "statsmodels GLM cov_type='HC1' omits n/(n-k) finite-sample correction (returns HC0); R true HC1 is sqrt(n/(n-k)) larger (~0.5%)")

source("tools/oracle/lib.R")
library(marginaleffects)
library(sandwich)

df <- read.csv("tests/oracle/data/oracle_main.csv")
fit <- lm(y_cont ~ treat + x1 + x2, data = df)
mod <- list(formula = "y_cont ~ treat + x1 + x2",
            family = "gaussian(identity)",
            fit_control = "lm()")

# predict overall
p <- avg_predictions(fit)
write_golden("ols_predict_overall_nonrobust", "oracle_main.csv", mod,
             "avg_predictions(fit)",
             list(coefficients = unname(coef(fit)),
                  estimate = p$estimate,
                  std_error = p$std.error,
                  conf_low = p$conf.low,
                  conf_high = p$conf.high))

# AME x1
s <- avg_slopes(fit, variables = "x1")
write_golden("ols_ame_x1_nonrobust", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1')",
             list(coefficients = unname(coef(fit)),
                  estimate = s$estimate,
                  std_error = s$std.error,
                  conf_low = s$conf.low,
                  conf_high = s$conf.high),
             labels = list("x1"))

# AME x1 HC1
sh <- avg_slopes(fit, variables = "x1", vcov = vcovHC(fit, type = "HC1"))
write_golden("ols_ame_x1_hc1", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', vcov = vcovHC(fit, type = 'HC1'))",
             list(coefficients = unname(coef(fit)),
                  estimate = sh$estimate,
                  std_error = sh$std.error,
                  conf_low = sh$conf.low,
                  conf_high = sh$conf.high),
             vcov = "HC1", labels = list("x1"))

# AME x1 cluster(g)
sc <- avg_slopes(fit, variables = "x1", vcov = vcovCL(fit, cluster = ~g, type = "HC1"))
write_golden("ols_ame_x1_cluster", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', vcov = vcovCL(fit, cluster = ~g, type = 'HC1'))",
             list(coefficients = unname(coef(fit)),
                  estimate = sc$estimate,
                  std_error = sc$std.error,
                  conf_low = sc$conf.low,
                  conf_high = sc$conf.high,
                  vcov_matrix = as.vector(vcovCL(fit, cluster = ~g, type = "HC1"))),
             vcov = "cluster(g)", labels = list("x1"),
             tolerances = list(std_error = 2e-5),
             notes = "alignment golden: vcovCL(cluster=~g, type='HC1') matrix included; OLS closed-form matches within machine precision")

# contrast treat
cmp <- avg_comparisons(fit, variables = "treat")
write_golden("ols_contrast_treat_nonrobust", "oracle_main.csv", mod,
             "avg_comparisons(fit, variables = 'treat')",
             list(coefficients = unname(coef(fit)),
                  estimate = cmp$estimate,
                  std_error = cmp$std.error,
                  conf_low = cmp$conf.low,
                  conf_high = cmp$conf.high),
             labels = list("treat"))

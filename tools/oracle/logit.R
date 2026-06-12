source("tools/oracle/lib.R")
library(marginaleffects)
library(sandwich)

df <- read.csv("tests/oracle/data/oracle_main.csv")
fit <- glm(y_bin ~ treat + x1 + x2, family = binomial(), data = df,
           control = glm.control(epsilon = 1e-12, maxit = 200))
mod <- list(formula = "y_bin ~ treat + x1 + x2",
            family = "binomial(logit)",
            fit_control = "glm.control(epsilon = 1e-12, maxit = 200)")

# predict overall
p <- avg_predictions(fit)
write_golden("logit_predict_overall_nonrobust", "oracle_main.csv", mod,
             "avg_predictions(fit)",
             list(coefficients = unname(coef(fit)),
                  estimate = p$estimate,
                  std_error = p$std.error,
                  conf_low = p$conf.low,
                  conf_high = p$conf.high))

# predict at treat=1 (counterfactual)
g1 <- avg_predictions(fit, newdata = datagrid(treat = 1, grid_type = "counterfactual"))
write_golden("logit_predict_at_treat1_nonrobust", "oracle_main.csv", mod,
             "avg_predictions(fit, newdata = datagrid(treat = 1, grid_type = 'counterfactual'))",
             list(coefficients = unname(coef(fit)),
                  estimate = g1$estimate,
                  std_error = g1$std.error,
                  conf_low = g1$conf.low,
                  conf_high = g1$conf.high))

# AME x1
s <- avg_slopes(fit, variables = "x1")
write_golden("logit_ame_x1_nonrobust", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1')",
             list(coefficients = unname(coef(fit)),
                  estimate = s$estimate,
                  std_error = s$std.error,
                  conf_low = s$conf.low,
                  conf_high = s$conf.high),
             labels = list("x1"))

# AME x1 with HC1
sh <- avg_slopes(fit, variables = "x1", vcov = vcovHC(fit, type = "HC1"))
write_golden("logit_ame_x1_hc1", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', vcov = vcovHC(fit, type = 'HC1'))",
             list(coefficients = unname(coef(fit)),
                  estimate = sh$estimate,
                  std_error = sh$std.error,
                  conf_low = sh$conf.low,
                  conf_high = sh$conf.high),
             vcov = "HC1", labels = list("x1"),
             tolerances = list(std_error = 0.006),
             notes = "statsmodels GLM cov_type='HC1' omits n/(n-k) finite-sample correction (returns HC0); R true HC1 is sqrt(n/(n-k)) larger (~0.5%)")

# AME x1 cluster(g)
sc <- avg_slopes(fit, variables = "x1", vcov = vcovCL(fit, cluster = ~g, type = "HC1"))
write_golden("logit_ame_x1_cluster", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', vcov = vcovCL(fit, cluster = ~g, type = 'HC1'))",
             list(coefficients = unname(coef(fit)),
                  estimate = sc$estimate,
                  std_error = sc$std.error,
                  conf_low = sc$conf.low,
                  conf_high = sc$conf.high,
                  vcov_matrix = as.vector(vcovCL(fit, cluster = ~g, type = "HC1"))),
             vcov = "cluster(g)", labels = list("x1"),
             tolerances = list(std_error = 2e-5),
             notes = "alignment golden: vcovCL(cluster=~g, type='HC1') matrix included; IRLS meat precision leaves SE ~1e-5 from vcov_matrix-derived SE")

# contrast treat
cmp <- avg_comparisons(fit, variables = "treat")
write_golden("logit_contrast_treat_nonrobust", "oracle_main.csv", mod,
             "avg_comparisons(fit, variables = 'treat')",
             list(coefficients = unname(coef(fit)),
                  estimate = cmp$estimate,
                  std_error = cmp$std.error,
                  conf_low = cmp$conf.low,
                  conf_high = cmp$conf.high),
             labels = list("treat"))

# weighted AME x1
sw <- avg_slopes(fit, variables = "x1", wts = "w")
write_golden("logit_ame_x1_weighted_nonrobust", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1', wts = 'w')",
             list(coefficients = unname(coef(fit)),
                  estimate = sw$estimate,
                  std_error = sw$std.error,
                  conf_low = sw$conf.low,
                  conf_high = sw$conf.high),
             labels = list("x1"))

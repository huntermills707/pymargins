source("tools/oracle/lib.R")
library(marginaleffects)
library(survey)

df <- read.csv("tests/oracle/data/oracle_main.csv")
design <- svydesign(ids = ~psu, strata = ~strata, weights = ~w,
                    data = df, nest = TRUE)
fit <- svyglm(y_bin ~ treat + x1 + x2, design = design, family = quasibinomial())
mod <- list(formula = "y_bin ~ treat + x1 + x2",
            family = "quasibinomial(logit)",
            fit_control = "svyglm(..., family = quasibinomial())")

# predict overall
p <- avg_predictions(fit)
write_golden("survey_logit_predict_overall_linearized", "oracle_main.csv", mod,
             "avg_predictions(fit)",
             list(coefficients = unname(coef(fit)),
                  estimate = p$estimate,
                  std_error = p$std.error,
                  conf_low = p$conf.low,
                  conf_high = p$conf.high),
             vcov = "survey linearized")

# AME x1
s <- avg_slopes(fit, variables = "x1")
write_golden("survey_logit_ame_x1_linearized", "oracle_main.csv", mod,
             "avg_slopes(fit, variables = 'x1')",
             list(coefficients = unname(coef(fit)),
                  estimate = s$estimate,
                  std_error = s$std.error,
                  conf_low = s$conf.low,
                  conf_high = s$conf.high),
             vcov = "survey linearized", labels = list("x1"))

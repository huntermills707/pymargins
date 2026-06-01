# Generate numeric reference for survey correctness test.
# Run locally with: Rscript tests/_generate_survey_reference.R

library(survey)
library(marginaleffects)

df <- read.csv("tests/survey_fixture.csv")

# Complex survey design
des <- svydesign(
  ids = ~psu,
  strata = ~strat,
  weights = ~w,
  data = df,
  nest = TRUE
)

# Quasibinomial GLM (matches statsmodels Binomial)
fit <- svyglm(y ~ x, design = des, family = quasibinomial())

# Average marginal effect of x
sl <- avg_slopes(fit, variables = "x")

out <- data.frame(
  estimand = "dydx(x)",
  estimate = as.numeric(sl$estimate),
  std_error = as.numeric(sl$std.error)
)

write.csv(out, "tests/survey_reference.csv", row.names = FALSE)
print(out)

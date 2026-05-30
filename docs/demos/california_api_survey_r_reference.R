# California API survey reference — R validation script
# Computes weighted AMEs and design-based SEs using survey + marginaleffects
# for cross-check against pymargins.

library(survey)
library(marginaleffects)

# Load the stratified sample (same CSV used by the Python demo)
apistrat <- read.csv("data/apistrat.csv")

# Declare the survey design
des <- svydesign(
  id = ~1,           # no clustering in apistrat
  strata = ~stype,
  weights = ~pw,
  fpc = ~fpc,
  data = apistrat
)

# Fit a weighted model (svyglm uses survey weights in estimation)
fit <- svyglm(api00 ~ meals + ell + avg.ed + mobility, design = des)

# Average marginal effect of meals
ame_meals <- avg_slopes(fit, variables = "meals")
print("AME of meals:")
print(ame_meals)

# Predicted API at meals = 50 (population-weighted)
pred_meals <- avg_predictions(fit, newdata = datagrid(meals = 50))
print("\nPredicted API at meals = 50:")
print(pred_meals)

# Contrast: ell = 5 vs ell = 35
contrast_ell <- avg_comparisons(fit, variables = list(ell = c(5, 35)))
print("\nContrast ell=5 vs ell=35:")
print(contrast_ell)

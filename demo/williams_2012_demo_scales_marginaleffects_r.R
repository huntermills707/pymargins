# ============================================================================
# R reference — scales / ratios / lifts via marginaleffects
# ============================================================================
library(marginaleffects)
set.seed(42)
n <- 5000
female <- rbinom(n, 1, 0.52)
black  <- rbinom(n, 1, 0.11)
age    <- sample(20:74, n, replace = TRUE)
agegrp <- as.numeric(cut(age, breaks = c(19,29,39,49,59,69,100)))
bmi    <- 22 + 0.15*age + 1.5*female + rnorm(n, 0, 4)
bmi    <- pmax(15, pmin(50, bmi))
lp     <- -4.0 + 0.55*black + 0.10*female + 0.06*age + 0.03*bmi +
          0.5*(agegrp==2) + 0.9*(agegrp==3) + 1.4*(agegrp==4) +
          2.0*(agegrp==5) + 2.6*(agegrp==6)
diabetes <- rbinom(n, 1, plogis(lp))
bp     <- 110 + 0.4*age + 2.5*black + 1.2*female + 0.5*bmi + rnorm(n, 0, 8)

# Do NOT convert to factor here; use numeric and let glm convert
df <- data.frame(diabetes, bp, black, female, age, agegrp, bmi)

# logistic fit (glm converts factor(black) internally)
fit_logit <- glm(diabetes ~ factor(black) + factor(female) + factor(agegrp) +
                   bmi + age,
                 data = df, family = binomial(link = "logit"))

cat("\n===== 1. RISK RATIO (lnratioavg with exp transform) =====\n")
rr <- avg_comparisons(fit_logit,
                      variables = "black",
                      comparison = "lnratioavg",
                      transform = exp)
print(rr)

rr_f <- avg_comparisons(fit_logit,
                        variables = "female",
                        comparison = "lnratioavg",
                        transform = exp)
print(rr_f)

cat("\n===== 2. TRUE LIFT (liftavg) =====\n")
lift <- avg_comparisons(fit_logit,
                        variables = "black",
                        comparison = "liftavg")
print(lift)

lift_f <- avg_comparisons(fit_logit,
                          variables = "female",
                          comparison = "liftavg")
print(lift_f)

cat("\n===== 3. MEM (atmeans) continuous slope =====\n")
mem <- slopes(fit_logit, newdata = datagrid(black=0, female=1,
                                             agegrp=3, bmi=27.5, age=47.5))
print(mem)

cat("\n===== 4. AME continuous slope =====\n")
# avg_slopes returns all slopes; we filter to `age` afterward
ame <- avg_slopes(fit_logit)
print(ame)

cat("\n===== 5. OLS continuous slope =====\n")
fit_ols <- lm(bp ~ factor(black) + factor(female) + factor(agegrp) +
                bmi + age, data = df)
mem_ols <- slopes(fit_ols, newdata = datagrid(black=0, female=1,
                                               agegrp=3, bmi=27.5, age=47.5))
print(mem_ols)

ame_ols <- avg_slopes(fit_ols)
print(ame_ols)

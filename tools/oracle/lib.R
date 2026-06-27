library(jsonlite)

write_golden <- function(case_id,
                         data,
                         model,
                         r_call,
                         quantities,
                         vcov = "nonrobust",
                         ci_convention = list(dist = "z", level = 0.95),
                         labels = NULL,
                         tolerances = setNames(list(), character(0)),
                         notes = "") {
  pkgs <- c("marginaleffects", "sandwich", "survey")
  pkgs <- pkgs[vapply(pkgs, requireNamespace, TRUE, quietly = TRUE)]
  payload <- list(
    case_id = case_id,
    created = format(Sys.Date()),
    r_version = R.version.string,
    packages = setNames(lapply(pkgs, function(p) as.character(packageVersion(p))), pkgs),
    data = data,
    model = model,
    r_call = r_call,
    vcov = vcov,
    ci_convention = ci_convention,
    labels = labels,
    quantities = quantities,
    tolerances = tolerances,
    notes = notes
  )
  path <- file.path("tests", "oracle", "golden", paste0(case_id, ".json"))
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  write(toJSON(payload, digits = NA, auto_unbox = TRUE, pretty = TRUE), path)
  cat("wrote", path, "\n")
}

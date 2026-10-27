#!/usr/bin/env Rscript
# Manual equal-weight ATT(g,t) estimator for T17 (staggered DiD) — R reference.
#
# Computes the same long-difference Callaway-Sant'Anna-style ATT(g, t) used in
# manual_python.py: for each treated cohort g in {4, 6, 8} and each post-period
# year t >= g, ATT(g, t) is
#
#     [E(y_did | cohort=g, year=t)    - E(y_did | cohort=g, year=g-1)]
#   - [E(y_did | cohort=-1, year=t)   - E(y_did | cohort=-1, year=g-1)]
#
# The reference estimand is the uniform average over the |S| = 12 supported
# (g, t) cells.

suppressPackageStartupMessages({
  library(jsonlite)
})

# Resolve data path. Prefer a script-relative path; fall back to a repo-root
# path when CWD is the repo root (e.g., Rscript run from there).
script_path <- tryCatch(
  normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
  error = function(e) NA_character_
)
if (!is.na(script_path) && nzchar(script_path) && file.exists(script_path)) {
  HERE <- dirname(script_path)
  data_path <- file.path(HERE, "..", "..", "data", "pilot_v3.csv")
} else {
  data_path <- "data/benchmark/pilot_v3.csv"
}

df <- read.csv(data_path)

COHORTS <- c(4, 6, 8)
LAST_YEAR <- 9
CONTROL_COHORT <- -1

att_gt <- function(df, g, t) {
  base <- g - 1
  treated_t    <- mean(df$y_did[df$cohort == g & df$year == t])
  treated_base <- mean(df$y_did[df$cohort == g & df$year == base])
  control_t    <- mean(df$y_did[df$cohort == CONTROL_COHORT & df$year == t])
  control_base <- mean(df$y_did[df$cohort == CONTROL_COHORT & df$year == base])
  (treated_t - treated_base) - (control_t - control_base)
}

cells <- list()
for (g in COHORTS) {
  for (t in g:LAST_YEAR) {
    cells[[length(cells) + 1L]] <- list(g = g, t = t, att_gt = att_gt(df, g, t))
  }
}

att_uniform <- mean(vapply(cells, function(c) c$att_gt, numeric(1)))

result <- list(
  platform = "r",
  estimand = "ATT_uniform = (1/|S|) * sum_{(g,t) in S} ATT(g,t)",
  n_cells = length(cells),
  cells = cells,
  att_uniform = att_uniform,
  population_truth = 20 / 12
)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE), "\n")

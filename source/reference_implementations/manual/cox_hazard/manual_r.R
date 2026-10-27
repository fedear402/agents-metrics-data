#!/usr/bin/env Rscript
# Manual reference for T13 (Cox PH) — R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
})

script_path <- tryCatch(
  normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
  error = function(e) NA_character_
)
data_path <- if (!is.na(script_path) && file.exists(script_path)) {
  file.path(dirname(script_path), "..", "..", "..", "data", "benchmark", "pilot_v4_cox.csv")
} else {
  "data/benchmark/pilot_v4_cox.csv"
}

df <- read.csv(data_path)
fit <- coxph(Surv(t_obs, event) ~ treat + x, data = df, ties = "breslow")
co <- summary(fit)$coefficients   # rows: treat, x; cols: coef, exp(coef), se(coef), z, Pr(>|z|)
out <- list(
  platform = "r",
  coef_treat = as.numeric(co["treat", "coef"]),
  se_treat   = as.numeric(co["treat", "se(coef)"]),
  coef_x     = as.numeric(co["x", "coef"]),
  se_x       = as.numeric(co["x", "se(coef)"]),
  events     = sum(df$event),
  n_obs      = nrow(df)
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

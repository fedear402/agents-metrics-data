#!/usr/bin/env Rscript
# Manual reference for T13 (Cox PH) — R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/cox_hazard"
data_path <- file.path(input_dir, "pilot_v4_cox.csv")

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
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

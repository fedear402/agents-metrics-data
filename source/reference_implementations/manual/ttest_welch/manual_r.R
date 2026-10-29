#!/usr/bin/env Rscript
# Manual reference for T07 (Welch two-sample t-test) — R.
suppressPackageStartupMessages({ library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/ttest_welch"
data_path <- file.path(input_dir, "pilot_v5_ttest.csv")
df <- read.csv(data_path)

a <- df$y[df$group == 0]; b <- df$y[df$group == 1]
tt <- t.test(b, a, var.equal = FALSE)   # Welch's t; compare group 1 vs group 0 so signs match Python
out <- list(
  platform = "r",
  t = as.numeric(tt$statistic),
  df = as.numeric(tt$parameter),
  p_value = as.numeric(tt$p.value),
  n_a = length(a),
  n_b = length(b)
)
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

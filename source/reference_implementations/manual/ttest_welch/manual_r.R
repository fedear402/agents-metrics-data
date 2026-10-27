#!/usr/bin/env Rscript
# Manual reference for T07 (Welch two-sample t-test) — R.
suppressPackageStartupMessages({ library(jsonlite) })
sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_ttest.csv") else "data/benchmark/pilot_v5_ttest.csv"
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
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

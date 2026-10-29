#!/usr/bin/env Rscript
# Manual reference for T06 (Cook's distance top-5) — R.

suppressPackageStartupMessages({ library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/cooks_distance"
data_path <- file.path(input_dir, "pilot_v5_collinear.csv")
df <- read.csv(data_path)

fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = df)
cd <- cooks.distance(fit)
ord <- order(-cd)[1:5]
out <- list(
  platform = "r",
  top5_obs_ids = as.integer(ord - 1L),    # zero-based to match Python
  top5_cooks_d = as.numeric(cd[ord]),
  n_obs = nrow(df)
)
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

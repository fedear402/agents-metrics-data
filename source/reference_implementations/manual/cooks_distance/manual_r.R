#!/usr/bin/env Rscript
# Manual reference for T06 (Cook's distance top-5) — R.

suppressPackageStartupMessages({ library(jsonlite) })

sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_collinear.csv") else "data/benchmark/pilot_v5_collinear.csv"
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
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

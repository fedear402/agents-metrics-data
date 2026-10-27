#!/usr/bin/env Rscript
# Manual reference for T18 (sharp RD, local-linear, triangular kernel, h=0.5) — R.

suppressPackageStartupMessages({ library(jsonlite) })
sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_rdd.csv") else "data/benchmark/pilot_v5_rdd.csv"
df <- read.csv(data_path)

h <- 0.5
inb <- abs(df$z) < h
z <- df$z[inb]; y <- df$y[inb]
w <- 1 - abs(z) / h

wls_intercept <- function(zs, ys, ws) {
  fit <- lm(ys ~ zs, weights = ws)
  as.numeric(coef(fit)[1])
}
mu_r <- wls_intercept(z[z >= 0], y[z >= 0], w[z >= 0])
mu_l <- wls_intercept(z[z <  0], y[z <  0], w[z <  0])
late <- mu_r - mu_l
out <- list(
  platform = "r",
  late = late,
  bandwidth = h,
  n_inside_band = sum(inb),
  n_file_rows = nrow(df),
  # n_obs is the effective estimation sample (observations inside the bandwidth),
  # matching the benchmark result schema and what models report.
  n_obs = sum(inb)
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

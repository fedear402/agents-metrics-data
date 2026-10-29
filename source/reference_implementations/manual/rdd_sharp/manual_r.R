#!/usr/bin/env Rscript
# Manual reference for T18 (sharp RD, local-linear, triangular kernel, h=0.5) — R.

suppressPackageStartupMessages({ library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/rdd_sharp"
data_path <- file.path(input_dir, "pilot_v5_rdd.csv")
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
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

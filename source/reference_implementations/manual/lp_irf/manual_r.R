#!/usr/bin/env Rscript
# Manual reference for T19 (local-projection IRF at h=4, NW SE) — R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
  library(lmtest)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/lp_irf"
data_path <- file.path(input_dir, "pilot_v5_lp.csv")
df <- read.csv(data_path); df <- df[order(df$t), ]

h <- 4L
T <- nrow(df)
yhp <- df$y[(h + 2):T]                  # y_{t+h} for t = 2..T-h (1-based)
shock_t <- df$shock[2:(T - h)]
y_lag <- df$y[1:(T - h - 1)]
fit <- lm(yhp ~ shock_t + y_lag)
nw <- NeweyWest(fit, lag = h, prewhite = FALSE, adjust = TRUE)
ct <- coeftest(fit, vcov = nw)
out <- list(
  platform = "r",
  irf_h4 = as.numeric(ct["shock_t", "Estimate"]),
  se_irf_h4 = as.numeric(ct["shock_t", "Std. Error"]),
  horizon = h,
  n_used = length(yhp)
)
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

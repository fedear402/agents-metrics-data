#!/usr/bin/env Rscript
# Manual reference for T22 (Granger F-test, y2 does not Granger-cause y1, 2 lags) — R.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/var_granger"
data_path <- file.path(input_dir, "pilot_v4_var.csv")

df <- read.csv(data_path)
n <- nrow(df)
idx <- 3:n  # use t = 3..n with two lags
Y     <- df$y1[idx]
lag1y1 <- df$y1[idx - 1]; lag2y1 <- df$y1[idx - 2]
lag1y2 <- df$y2[idx - 1]; lag2y2 <- df$y2[idx - 2]

fit_res <- lm(Y ~ lag1y1 + lag2y1)
fit_unr <- lm(Y ~ lag1y1 + lag2y1 + lag1y2 + lag2y2)
rss_res <- sum(resid(fit_res) ^ 2)
rss_unr <- sum(resid(fit_unr) ^ 2)
q <- 2
n_used <- length(Y)
k_unr <- length(coef(fit_unr))
F  <- ((rss_res - rss_unr) / q) / (rss_unr / (n_used - k_unr))
pv <- 1 - pf(F, q, n_used - k_unr)

out <- list(
  platform = "r",
  F = F,
  df_num = q,
  df_den = n_used - k_unr,
  p_value = pv,
  n_obs = n_used
)
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

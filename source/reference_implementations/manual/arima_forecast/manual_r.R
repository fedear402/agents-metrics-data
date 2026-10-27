#!/usr/bin/env Rscript
# Manual reference for T02 (ARIMA(1,1,1) one-step forecast) — R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(forecast)
})

script_path <- tryCatch(
  normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
  error = function(e) NA_character_
)
data_path <- if (!is.na(script_path) && file.exists(script_path)) {
  file.path(dirname(script_path), "..", "..", "data", "pilot_v4_arima.csv")
} else {
  "data/benchmark/pilot_v4_arima.csv"
}

df <- read.csv(data_path)
fit <- Arima(df$y, order = c(1, 1, 1), method = "ML")
fc <- forecast(fit, h = 1, level = 95)
out <- list(
  platform = "r",
  forecast = as.numeric(fc$mean[1]),
  se = as.numeric((fc$upper[1, "95%"] - fc$lower[1, "95%"]) / (2 * qnorm(0.975))),
  lower_95 = as.numeric(fc$lower[1, "95%"]),
  upper_95 = as.numeric(fc$upper[1, "95%"]),
  n_obs = nrow(df)
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

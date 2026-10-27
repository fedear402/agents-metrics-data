#!/usr/bin/env Rscript
# Manual reference for T16 (Arellano-Bond) — R via the plm package's pgmm.

suppressPackageStartupMessages({
  library(jsonlite)
  library(plm)
})

sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_dynpanel.csv") else "data/benchmark/pilot_v5_dynpanel.csv"
df <- read.csv(data_path)

pd <- pdata.frame(df, index = c("firm", "year"))
fit <- pgmm(y ~ lag(y, 1) + x | lag(y, 2:99),
            data = pd, effect = "individual", model = "onestep",
            transformation = "d")
co <- coef(fit)
sefit <- summary(fit)$coefficients[, "Std. Error"]
out <- list(
  platform = "r",
  rho = as.numeric(co["lag(y, 1)"]),
  se_rho = as.numeric(sefit["lag(y, 1)"]),
  beta_x = as.numeric(co["x"]),
  se_beta_x = as.numeric(sefit["x"]),
  n_obs = length(residuals(fit)),
  n_firms = length(unique(df$firm))
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

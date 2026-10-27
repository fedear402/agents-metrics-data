#!/usr/bin/env Rscript
# Manual reference for T08 (one-way ANOVA) — R.
suppressPackageStartupMessages({ library(jsonlite) })
sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_anova.csv") else "data/benchmark/pilot_v5_anova.csv"
df <- read.csv(data_path)
df$group <- as.factor(df$group)
fit <- aov(y ~ group, data = df)
s <- summary(fit)[[1]]
out <- list(
  platform = "r",
  F = as.numeric(s["group", "F value"]),
  df_num = as.integer(s["group", "Df"]),
  df_den = as.integer(s["Residuals", "Df"]),
  p_value = as.numeric(s["group", "Pr(>F)"]),
  n_obs = nrow(df)
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

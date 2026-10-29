#!/usr/bin/env Rscript
# Manual reference for T08 (one-way ANOVA) — R.
suppressPackageStartupMessages({ library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/anova_oneway"
data_path <- file.path(input_dir, "pilot_v5_anova.csv")
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
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

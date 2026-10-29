#!/usr/bin/env Rscript
# Manual reference for T09 (chi-square test of independence) — R.
suppressPackageStartupMessages({ library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/chisq_indep"
data_path <- file.path(input_dir, "pilot_v5_chisq.csv")
df <- read.csv(data_path)
ct <- table(df$row, df$col)
res <- chisq.test(ct, correct = FALSE)
out <- list(platform = "r", chi2 = as.numeric(res$statistic), df = as.integer(res$parameter), p_value = as.numeric(res$p.value), n_obs = nrow(df))
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

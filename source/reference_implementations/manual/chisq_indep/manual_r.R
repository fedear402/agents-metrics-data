#!/usr/bin/env Rscript
# Manual reference for T09 (chi-square test of independence) — R.
suppressPackageStartupMessages({ library(jsonlite) })
sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_chisq.csv") else "data/benchmark/pilot_v5_chisq.csv"
df <- read.csv(data_path)
ct <- table(df$row, df$col)
res <- chisq.test(ct, correct = FALSE)
out <- list(platform = "r", chi2 = as.numeric(res$statistic), df = as.integer(res$parameter), p_value = as.numeric(res$p.value), n_obs = nrow(df))
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

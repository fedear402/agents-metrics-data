#!/usr/bin/env Rscript
# Manual reference for T05 (VIF) — R.

suppressPackageStartupMessages({ library(jsonlite) })

sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_collinear.csv") else "data/benchmark/pilot_v5_collinear.csv"
df <- read.csv(data_path)

cols <- c("x1", "x2", "x3", "x4", "x5")
vif <- numeric(length(cols))
names(vif) <- cols
for (c in cols) {
  others <- setdiff(cols, c)
  f <- as.formula(paste0(c, " ~ ", paste(others, collapse = " + ")))
  fit <- lm(f, data = df)
  r2 <- summary(fit)$r.squared
  vif[c] <- 1 / (1 - r2)
}
out <- c(list(platform = "r"), as.list(setNames(vif, paste0("vif_", names(vif)))), list(n_obs = nrow(df)))
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

#!/usr/bin/env Rscript
# Manual reference for T21 (VAR(2) orthogonal IRF, y1 response to y2 shock, h=4) — R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(vars)
})

script_path <- tryCatch(
  normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])),
  error = function(e) NA_character_
)
data_path <- if (!is.na(script_path) && file.exists(script_path)) {
  file.path(dirname(script_path), "..", "..", "data", "pilot_v4_var.csv")
} else {
  "data/benchmark/pilot_v4_var.csv"
}

df <- read.csv(data_path)
Y <- as.matrix(df[, c("y1", "y2")])
fit <- VAR(Y, p = 2, type = "const")
ir <- irf(fit, impulse = "y2", response = "y1", n.ahead = 10, ortho = TRUE,
          boot = FALSE, cumulative = FALSE)
# ir$irf$y2 is a (n.ahead + 1) x 1 matrix; row 1 is h = 0, row 5 is h = 4.
val <- as.numeric(ir$irf$y2[5, "y1"])
out <- list(
  platform = "r",
  irf_y1_to_y2_h4 = val,
  lag_order = 2,
  identification = "Cholesky, y1 ordered first",
  n_obs = nrow(df)
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

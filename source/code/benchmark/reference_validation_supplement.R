#!/usr/bin/env Rscript
# Supplementary R routes for the reference-validation table.
#
# The stored manual programs check these four references in Stata only, or in R
# with a different convention. Each route below re-estimates the reference
# convention in R from the catalog input.
#
# Usage: Rscript reference_validation_supplement.R <task_id> <input.csv>
# Output: one JSON object on stdout. Keys are catalog terms; "<term>:se" keys
# hold standard errors.

suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
})

args <- commandArgs(trailingOnly = TRUE)
task <- args[1]
d <- read.csv(args[2])
out <- list()

coef_se <- function(fit, V, names) {
  b <- coef(fit)
  s <- sqrt(diag(V))
  for (j in seq_along(names)) {
    out[[names[j]]] <<- unname(b[j])
    out[[paste0(names[j], ":se")]] <<- unname(s[j])
  }
}

if (task == "ols_robust") {
  # HC1: sandwich with the n / (n - k) factor.
  fit <- lm(y_lin ~ x1 + x2, data = d)
  coef_se(fit, vcovHC(fit, type = "HC1"), c("const", "x1", "x2"))
} else if (task == "cluster_ols") {
  # CR1: G / (G - 1) times (n - 1) / (n - k).
  fit <- lm(y_panel ~ x1 + x2, data = d)
  coef_se(fit, vcovCL(fit, cluster = ~cluster, type = "HC1", cadjust = TRUE), c("const", "x1", "x2"))
} else if (task == "two_way_cluster") {
  # Cameron-Gelbach-Miller: firm + year - intersection, CR1 factor in each component.
  fit <- lm(y_twocluster ~ x, data = d)
  V <- vcovCL(fit, cluster = ~ firm + year, type = "HC1", cadjust = TRUE, multi0 = FALSE)
  coef_se(fit, V, c("const", "x"))
} else if (task == "arima_forecast") {
  # Exact ML ARIMA(1,1,1) without drift; normal 95 percent interval from the ML innovation variance.
  d <- d[order(d$t), ]
  fit <- arima(d$y, order = c(1, 1, 1), method = "ML")
  p <- predict(fit, n.ahead = 1)
  z <- qnorm(0.975)
  out$forecast <- as.numeric(p$pred[1])
  out[["forecast:se"]] <- as.numeric(p$se[1])
  out$lower_95 <- out$forecast - z * out[["forecast:se"]]
  out$upper_95 <- out$forecast + z * out[["forecast:se"]]
} else {
  stop(paste("no supplementary route for", task))
}

cat(toJSON(out, digits = NA, auto_unbox = TRUE), "\n")

#!/usr/bin/env Rscript
# Manual reference for T14 (two-way FE OLS with cluster-robust SE on firm) — R.
# Uses a within-style estimator by including i.firm and i.year dummies in lm,
# then computing CR1 cluster-robust SE on firm.

suppressPackageStartupMessages({ library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/twoway_fe"
data_path <- file.path(input_dir, "pilot_v3.csv")
df <- read.csv(data_path)

df$firm <- as.factor(df$firm)
df$year <- as.factor(df$year)
fit <- lm(y_twocluster ~ x + firm + year, data = df)
b <- coef(fit)["x"]
resid <- residuals(fit)

# CR1 cluster-robust SE on firm, computed manually.
X <- model.matrix(fit)
n <- nrow(X); K <- ncol(X)
XtX_inv <- solve(crossprod(X))
firms <- unique(df$firm)
G <- length(firms)
meat <- matrix(0, K, K)
for (g in firms) {
  mask <- df$firm == g
  Xg <- X[mask, , drop = FALSE]; eg <- resid[mask]
  ug <- as.vector(crossprod(Xg, eg))
  meat <- meat + tcrossprod(ug)
}
correction <- (G / (G - 1)) * ((n - 1) / (n - K))
V <- correction * XtX_inv %*% meat %*% XtX_inv
se_x <- sqrt(V["x", "x"])
out <- list(platform = "r", x = as.numeric(b), se_x_cluster_firm = as.numeric(se_x), n_obs = n, n_firms = G, K_absorbed = K)
cat(toJSON(out, digits = NA, pretty = TRUE, auto_unbox = TRUE), "\n")

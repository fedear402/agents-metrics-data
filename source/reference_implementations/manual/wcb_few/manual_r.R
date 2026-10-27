#!/usr/bin/env Rscript
# Manual reference for T20 (wild cluster bootstrap, 5 clusters, B=999) — R.
# Pure base-R implementation (no fwildclusterboot dependency) so the reference
# matches the Python manual implementation algorithmically.

suppressPackageStartupMessages({ library(jsonlite) })
sp <- tryCatch(normalizePath(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), error = function(e) NA_character_)
data_path <- if (!is.na(sp) && file.exists(sp)) file.path(dirname(sp), "..", "..", "..", "data", "benchmark", "pilot_v5_wcb.csv") else "data/benchmark/pilot_v5_wcb.csv"
df <- read.csv(data_path)

y <- df$y; x <- df$x; cluster <- df$cluster
n <- length(y)
X <- cbind(1, x); K <- ncol(X)
XtX_inv <- solve(crossprod(X))
beta <- as.vector(XtX_inv %*% crossprod(X, y))
resid <- y - X %*% beta
clusters <- sort(unique(cluster))
G <- length(clusters)

cr1_se <- function(resid_e, X_e) {
  meat <- matrix(0, K, K)
  for (g in clusters) {
    mask <- cluster == g
    Xg <- X_e[mask, , drop = FALSE]; eg <- resid_e[mask]
    ug <- as.vector(crossprod(Xg, eg))
    meat <- meat + tcrossprod(ug)
  }
  correction <- (G / (G - 1)) * ((n - 1) / (n - K))
  V <- correction * XtX_inv %*% meat %*% XtX_inv
  sqrt(diag(V))
}
se <- cr1_se(as.vector(resid), X)
t_obs <- beta[2] / se[2]

# Restricted (H0: beta_x = 0) residuals
beta_r <- c(mean(y), 0)
resid_r <- y - X %*% beta_r
set.seed(20260520L)
B <- 999L
extreme <- 0L
for (b in seq_len(B)) {
  w <- sample(c(-1, 1), G, replace = TRUE)
  wf <- w[match(cluster, clusters)]
  ystar <- X %*% beta_r + as.vector(resid_r) * wf
  beta_star <- as.vector(XtX_inv %*% crossprod(X, ystar))
  resid_star <- ystar - X %*% beta_star
  se_star <- cr1_se(as.vector(resid_star), X)
  t_star <- beta_star[2] / se_star[2]
  if (abs(t_star) >= abs(t_obs)) extreme <- extreme + 1L
}
out <- list(
  platform = "r",
  x = as.numeric(beta[2]),
  se_x_cr1 = as.numeric(se[2]),
  t_obs = as.numeric(t_obs),
  p_wcb = extreme / B,
  B = B,
  n_clusters = G,
  n_obs = n
)
cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE), "\n")

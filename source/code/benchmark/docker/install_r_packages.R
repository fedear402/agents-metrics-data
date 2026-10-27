#!/usr/bin/env Rscript
# Install the R econometrics packages the agent needs inside the R3 container.
#
# Package set is derived from the R libraries actually used by generated agent
# scripts across the existing sweep (counts from `grep library(...)` over the
# run corpus, 2026-06-30):
#   jsonlite, sandwich, lmtest, vars, plm, AER, lfe, ivreg, forecast,
#   multiwayvcov, car, fixest.  `survival` ships with R base and is not installed
#   here.
#
# REPRODUCIBILITY: we install from a DATED Posit Public Package Manager (P3M)
# snapshot so the resolved package versions are frozen to a point in time rather
# than "whatever CRAN had on build day". Bump P3M_SNAPSHOT deliberately and
# rebuild if you need newer packages. The reference outputs were generated on the
# host R 4.6.0 stack; if exact numeric parity ever drifts, pin individual
# packages with remotes::install_version() to the host versions.

p3m_snapshot <- Sys.getenv("P3M_SNAPSHOT", "2026-06-01")
repo <- sprintf("https://packagemanager.posit.co/cran/%s", p3m_snapshot)
options(
  repos = c(P3M = repo),
  Ncpus = max(1L, parallel::detectCores()),
  warn = 2  # turn install warnings into errors so a failed build is loud
)

pkgs <- c(
  "jsonlite",     # result.json read/write — used by ~all R agent scripts
  "sandwich",     # robust / clustered covariance
  "lmtest",       # coeftest, robust inference
  "car",          # linearHypothesis, diagnostics
  "AER",          # ivreg + applied-econometrics helpers
  "ivreg",        # standalone IV regression
  "lfe",          # felm high-dimensional fixed effects
  "fixest",       # feols/feglm fast fixed effects
  "plm",          # panel-data models
  "multiwayvcov", # multiway clustered covariance
  "vars",         # vector autoregression
  "forecast"      # ARIMA / time-series forecasting
)

message(sprintf("Installing %d R packages from %s", length(pkgs), repo))
install.packages(pkgs)

# Hard fail if anything did not land, so `docker build` stops here rather than
# producing an image that silently lacks an estimator.
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("R package install incomplete; missing: ", paste(missing, collapse = ", "))
}
message("All R econometrics packages installed and loadable.")

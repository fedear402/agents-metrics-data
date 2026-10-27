args <- commandArgs(trailingOnly=TRUE)
folder <- args[1]
out <- list()
for (kind in c("target", "example")) {
  d <- read.csv(file.path(folder, paste0("lp_", kind, ".csv")))
  sample <- d
  horizons <- if (kind == "target") 0:2 else 0:1
  for (h in c(-1, horizons[horizons > 0])) {
    q <- d[c("unit", "period", "y")]
    q$period <- q$period - h
    names(q)[3] <- paste0("yh", if (h == -1) "lag" else h)
    sample <- merge(sample, q, by=c("unit", "period"))
  }
  C <- model.matrix(~ yhlag + factor(unit) + factor(period), sample)
  X <- cbind(x=sample$x, C)
  Z <- cbind(z=sample$z, C)
  PZX <- Z %*% qr.solve(Z, X)
  values <- sapply(horizons, function(h) {
    y <- if (h == 0) sample$y else sample[[paste0("yh", h)]]
    as.numeric(solve(crossprod(PZX, X), crossprod(PZX, y))[1])
  })
  write.table(data.frame(term=paste0("lp_h",horizons),estimate=values,n_obs=nrow(sample)),
              file.path(folder,paste0("r_lp_",kind,".csv")),row.names=FALSE,sep=",")
  d <- read.csv(file.path(folder, paste0("ppml_", kind, ".csv")))
  groups <- if (kind == "target") c("pair","exporter_year","importer_year") else c("pair","time")
  # These saved data separate through zero-only fixed-effect groups.
  # The Python preflight independently checks the general separation condition.
  keep <- rep(TRUE,nrow(d))
  repeat {
    before <- keep
    for (g in groups) {
      totals <- tapply(d$trade[keep],d[[g]][keep],sum)
      keep <- keep & !(d[[g]] %in% names(totals)[totals == 0])
    }
    if (identical(before,keep)) break
  }
  kept <- d[keep,]
  Z <- model.matrix(as.formula(paste("~",paste(paste0("factor(",groups,")"),collapse="+"))),kept)
  qrZ <- qr(Z)
  Z <- Z[,qrZ$pivot[seq_len(qrZ$rank)],drop=FALSE]
  X <- cbind(agreement=kept$agreement,Z)
  fit <- glm.fit(X,kept$trade,family=poisson(),control=glm.control(epsilon=1e-12,maxit=100))
  stopifnot(fit$converged)
  mu <- fit$fitted.values
  bread <- solve(crossprod(X,X*mu))
  scores <- rowsum(X*(kept$trade-mu),kept$pair)
  N <- nrow(X); K <- ncol(X); G <- nrow(scores)
  V <- bread %*% crossprod(scores) %*% bread * G/(G-1)*(N-1)/(N-K)
  values <- c(fit$coefficients[1],expm1(fit$coefficients[1]),sqrt(V[1,1]),N,sum(!keep),sum(kept$trade==0))
  terms <- c("agreement_coefficient","proportional_mean_change","pair_cluster_se","retained_observations","separated_observations","retained_zero_observations")
  write.table(data.frame(term=terms,estimate=values,n_obs=N),file.path(folder,paste0("r_ppml_",kind,".csv")),row.names=FALSE,sep=",")
}
cat("R_PREFLIGHT_COMPLETE\n")

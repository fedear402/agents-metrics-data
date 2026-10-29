# Native R check for lp04_ppml.
# Source: the lp04_ppml branch of the six-task native R check (six_native.R),
# which wrote native_r_target.json.
# Route: iterative removal of zero-only fixed-effect groups (the separated zeros
# of this design), pivoted QR for a full-rank dummy basis, and glm.fit Poisson.
#
# The scored terms are agreement_coefficient and proportional_mean_change.
# The pair-clustered standard error and the three observation counts of the
# original task are printed under the extra key "diagnostics".
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/lp04_ppml.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/lp04_ppml"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
d <- read(); keep <- rep(TRUE,nrow(d))
repeat {
 old <- keep
 for(v in c('pair','exporter_year','importer_year')) {
  sums <- tapply(d$trade[keep],d[[v]][keep],sum)
  zero <- names(sums)[sums==0]
  keep <- keep & !(as.character(d[[v]]) %in% zero)
 }
 if(identical(old,keep)) break
}
a <- d[keep,]; X0 <- model.matrix(~agreement+factor(pair)+factor(exporter_year)+factor(importer_year),a)
q <- qr(X0); X <- X0[,q$pivot[seq_len(q$rank)],drop=FALSE]
fit <- glm.fit(X,a$trade,family=poisson(),control=glm.control(epsilon=1e-12,maxit=200))
mu <- fit$fitted.values; bread <- solve(crossprod(X,mu*X))
scores <- rowsum(X*(a$trade-mu),a$pair)
N <- nrow(X); K <- ncol(X); G <- nrow(scores)
V <- bread %*% crossprod(scores) %*% bread*G/(G-1)*(N-1)/(N-K)
j <- match('agreement',colnames(X)); b <- fit$coefficients[j]
out <- c(agreement_coefficient=unname(b),proportional_mean_change=unname(expm1(b)),pair_cluster_se=sqrt(V[j,j]),retained_observations=N,separated_observations=sum(!keep),retained_zero_observations=sum(a$trade==0))
scored <- c('agreement_coefficient','proportional_mean_change')
result <- list(task_id = "lp04_ppml",
               estimates = lapply(scored, function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = N)),
               diagnostics = as.list(out[setdiff(names(out), scored)]))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

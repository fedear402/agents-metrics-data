# Reference program for hard_truncated_normal_effects (R).
# This is the native R check of the task package. It fits the left-truncated
# normal model by BFGS on the direct likelihood with the analytic score.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_truncated_normal_effects.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_truncated_normal_effects"
d <- read.csv(file.path(input_dir, "input.csv"))
X <- cbind(1,d$x,d$z)
objective <- function(b) {
 sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3])
 -sum(dnorm((d$y-mu)/sigma,log=TRUE)-b[4]-pnorm(mu/sigma,log.p=TRUE))
}
gradient <- function(b) {
 sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3]); a <- -mu/sigma; t <- (d$y-mu)/sigma
 lam <- exp(dnorm(a,log=TRUE)-pnorm(-a,log.p=TRUE))
 -c(crossprod(X,(t-lam)/sigma),sum(t*t-1-a*lam))
}
fit <- optim(c(0,.5,0,0),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0,max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3]); a <- -mu/sigma
lam <- exp(dnorm(a,log=TRUE)-pnorm(-a,log.p=TRUE))
out <- c(mean_observed=mean(mu+sigma*lam),ame_x=mean(b[2]*(1+a*lam-lam^2)),latent_x=b[2])
estimates <- lapply(names(out), function(term) list(term=term, estimate=unname(out[[term]]), se=NA, n_obs=nrow(d)))
result <- list(task_id="hard_truncated_normal_effects", platform="r", estimates=estimates)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

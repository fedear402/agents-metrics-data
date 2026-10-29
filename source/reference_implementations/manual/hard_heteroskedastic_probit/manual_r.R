# Reference program for hard_heteroskedastic_probit (R).
# This is the native R check of the task package. It fits the
# heteroskedastic probit by BFGS on the direct likelihood with the analytic score.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_heteroskedastic_probit.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_heteroskedastic_probit"
d <- read.csv(file.path(input_dir, "input.csv"))
X <- cbind(1,d$x,d$d); sign <- 2*d$y-1
objective <- function(b) -sum(pnorm(sign*as.vector(X%*%b[1:3])*exp(-b[4]*d$z),log.p=TRUE))
gradient <- function(b) {
 inv <- exp(-b[4]*d$z); eta <- as.vector(X%*%b[1:3])*inv; t <- sign*eta
 r <- sign*exp(dnorm(t,log=TRUE)-pnorm(t,log.p=TRUE))
 -c(crossprod(X,r*inv),sum(r*(-eta*d$z)))
}
fit <- optim(c(0,.4,-.3,.2),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0, max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; sigma <- exp(b[4]*d$z); mu <- as.vector(X%*%b[1:3])
out <- c(gamma_z=b[4],ame_x=mean(dnorm(mu/sigma)*b[2]/sigma),effect_d=mean(pnorm((b[1]+b[2]*d$x+b[3])/sigma)-pnorm((b[1]+b[2]*d$x)/sigma)))
estimates <- lapply(names(out), function(term) list(term=term, estimate=unname(out[[term]]), se=NA, n_obs=nrow(d)))
result <- list(task_id="hard_heteroskedastic_probit", platform="r", estimates=estimates)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

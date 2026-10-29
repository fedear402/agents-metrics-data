# Reference program for hard_poisson_hurdle_effects (R).
# This is the native R check of the task package. It fits the hurdle by glm logit
# and the zero-truncated Poisson part by BFGS on the direct likelihood.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_poisson_hurdle_effects.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_poisson_hurdle_effects"
d <- read.csv(file.path(input_dir, "input.csv"))
X <- cbind(1,d$x,d$d); positive <- d$y>0; XP <- X[positive,,drop=FALSE]; y <- d$y[positive]
fit_logit <- glm(positive~x+d,data=d,family=binomial(),control=glm.control(epsilon=1e-12,maxit=200))
stopifnot(fit_logit$converged)
a <- unname(coef(fit_logit))
objective <- function(b) {
 eta <- as.vector(XP%*%b); lam <- exp(eta)
 -sum(y*eta-lam-lgamma(y+1)-log(-expm1(-lam)))
}
gradient <- function(b) {lam <- exp(as.vector(XP%*%b)); -as.vector(crossprod(XP,y-lam/(-expm1(-lam))))}
fit <- optim(c(0,.2,.2),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0,max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; p <- plogis(as.vector(X%*%a)); lam <- exp(as.vector(X%*%b)); q <- -expm1(-lam)
mean_fun <- function(xx) {l <- exp(as.vector(xx%*%b)); plogis(as.vector(xx%*%a))*l/(-expm1(-l))}
one <- zero <- X; one[,3] <- 1; zero[,3] <- 0
out <- c(mean_y=mean(p*lam/q),ame_x=mean(p*(1-p)*a[2]*lam/q+p*b[2]*lam*(q-lam*exp(-lam))/q^2),effect_d=mean(mean_fun(one)-mean_fun(zero)),prob_at_least_three=mean(p*ppois(2,lam,lower.tail=FALSE)/q))
estimates <- lapply(names(out), function(term) list(term=term, estimate=unname(out[[term]]), se=NA, n_obs=nrow(d)))
result <- list(task_id="hard_poisson_hurdle_effects", platform="r", estimates=estimates)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

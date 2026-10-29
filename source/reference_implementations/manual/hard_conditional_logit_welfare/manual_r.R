# Reference program for hard_conditional_logit_welfare (R).
# This is the R native reference of the task package (welfare inputs). It
# maximizes the grouped conditional-logit likelihood with optim BFGS and an
# analytic score. Person weights enter only the two policy averages.
# Run from the data repository root: Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_conditional_logit_welfare.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_conditional_logit_welfare"

# Direct grouped conditional-logit likelihood. Person weights enter welfare only.
d <- read.csv(file.path(input_dir, "input.csv"))
d <- d[order(d$person, d$alternative), ]
stopifnot(all(table(d$person) == 3), all(tapply(d$chosen,d$person,sum) == 1))
X <- cbind(d$cost,d$quality,as.numeric(d$alternative==1),as.numeric(d$alternative==2))
n <- length(unique(d$person))
lse <- function(u) {m <- apply(u,1,max); m+log(rowSums(exp(u-m)))}
objective <- function(b) {
 u <- matrix(drop(X%*%b),ncol=3,byrow=TRUE)
 sum(lse(u))-sum(d$chosen*drop(X%*%b))
}
gradient <- function(b) {
 u <- matrix(drop(X%*%b),ncol=3,byrow=TRUE)
 p <- exp(u-lse(u))
 drop(crossprod(X,as.vector(t(p))-d$chosen))
}
fit <- optim(c(-1,.5,0,0),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=2000))
stopifnot(max(abs(gradient(fit$par)))<1e-4)
b <- fit$par
old <- matrix(drop(X%*%b),ncol=3,byrow=TRUE)
Xp <- X; Xp[d$alternative==1,1] <- .7*Xp[d$alternative==1,1]
policy <- matrix(drop(Xp%*%b),ncol=3,byrow=TRUE)
w <- d$weight[d$alternative==0]
gain <- (lse(policy)-lse(old))/(-b[1])
share <- exp(policy[,2]-lse(policy))-exp(old[,2]-lse(old))
values <- c(welfare_gain=weighted.mean(gain,w),share_change_alt1=weighted.mean(share,w),cost_coefficient=b[1])
out <- data.frame(term=names(values),estimate=unname(values),se=NA_real_,n_obs=n)
cat(jsonlite::toJSON(list(estimates=out), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

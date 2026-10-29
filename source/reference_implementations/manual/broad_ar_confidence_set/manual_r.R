# Reference program for broad_ar_confidence_set (R).
# This is the native R check of the task build. It computes the residual sums
# of squares with lm() at beta = -1, 0, 1. These three values fix the quadratic
# acceptance region of the Anderson-Rubin F test. polyroot() gives its roots.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_ar_confidence_set.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_ar_confidence_set"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
crit <- qf(.95,2,nrow(d)-4)
Q <- function(beta) {
  d$transformed <- d$y-beta*d$endog
  rr <- sum(resid(lm(transformed~x,data=d))^2)
  uu <- sum(resid(lm(transformed~x+z1+z2,data=d))^2)
  c(q=rr-uu-2*crit*uu/(nrow(d)-4),F=((rr-uu)/2)/(uu/(nrow(d)-4)))
}
q0 <- Q(0)[1];qp <- Q(1)[1];qm <- Q(-1)[1]
aa <- (qp+qm)/2-q0;bb <- (qp-qm)/2
roots <- sort(Re(polyroot(c(q0,bb,aa))))
add("boundary_low",roots[1]);add("boundary_high",roots[2])
add("accepts_zero",q0<=0);add("unbounded",aa<0);add("ar_f_at_one",Q(1)[2])
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

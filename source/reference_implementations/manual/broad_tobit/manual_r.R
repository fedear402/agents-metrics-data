# Reference program for broad_tobit (R).
# This is the native R check of the task build. survival::survreg fits the
# Gaussian model with left censoring at zero. sigma is the fitted scale.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_tobit.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_tobit"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
library(survival)
f <- survreg(Surv(y,y>0,type="left")~x+z,data=d,dist="gaussian",control=survreg.control(maxiter=100))
for(j in 1:3) add(c("const","x","z")[j],coef(f)[j],sqrt(vcov(f)[j,j]))
add("sigma",f$scale)
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

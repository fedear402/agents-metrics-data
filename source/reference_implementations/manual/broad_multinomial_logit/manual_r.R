# Reference program for broad_multinomial_logit (R).
# This is the native R check of the task build. nnet::multinom fits the
# multinomial logit with base outcome 0. Standard errors come from its Hessian.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_multinomial_logit.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_multinomial_logit"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
f <- nnet::multinom(factor(y,levels=0:2)~x+z,data=d,Hess=TRUE,trace=FALSE,maxit=2000,reltol=1e-12)
v <- summary(f)$standard.errors
for(k in 1:2) for(variable in c("x","z")) add(paste0(variable,"_cat",k),coef(f)[k,variable],v[k,variable])
add("prob_cat2_zero",predict(f,newdata=data.frame(x=0,z=0),type="probs")[3])
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

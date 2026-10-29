# Reference program for broad_ordered_logit (R).
# This is the native R check of the task build. MASS::polr fits the ordered
# logit. Standard errors come from its Hessian.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_ordered_logit.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_ordered_logit"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
f <- MASS::polr(ordered(y)~x+z,data=d,method="logistic",Hess=TRUE,control=list(reltol=1e-12,maxit=1000))
v <- vcov(f)
add("x",coef(f)["x"],sqrt(v["x","x"]))
add("z",coef(f)["z"],sqrt(v["z","z"]))
add("prob_high",predict(f,newdata=data.frame(x=1,z=0),type="probs")[3])
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

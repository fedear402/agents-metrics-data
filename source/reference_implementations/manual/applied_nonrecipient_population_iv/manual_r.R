# Native R check for applied_nonrecipient_population_iv.
# Source: the applied_nonrecipient_population_iv branch of the six-task native R
# check (six_native.R), which wrote native_r_target.json.
# Route: separate weighted projection 2SLS fits by eligibility group (qr), then
# the census-weighted mean of the two group effects.
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/applied_nonrecipient_population_iv.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/applied_nonrecipient_population_iv"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
iv <- function(y,C,x,z,w=rep(1,length(y))) {
  X <- cbind(C,x)*sqrt(w); Z <- cbind(C,z)*sqrt(w)
  fitted <- qr.fitted(qr(Z),X)
  as.numeric(qr.coef(qr(fitted),y*sqrt(w)))[ncol(X)]
}
d <- subset(read('households.csv'),recipient==0); census <- read('census.csv')
effect <- sapply(0:1,function(g) {
 a <- subset(d,eligible==g)
 iv(a$outcome,cbind(1,a$baseline),a$dose,a$instrument,a$weight)*weighted.mean(a$dose,a$weight)
})
out <- c(ineligible_effect=effect[1],eligible_nonrecipient_effect=effect[2],population_effect=weighted.mean(effect,census$nonrecipient_population[match(0:1,census$eligible)]))
n <- nrow(d)
result <- list(task_id = "applied_nonrecipient_population_iv",
               estimates = lapply(names(out), function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = n)))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

# Reference program for broad_oaxaca (R).
# This is the native R check of the task build. lm() fits each group, and the
# decomposition uses group 0 coefficients as the reference.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_oaxaca.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_oaxaca"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
a <- d[d$group==0,];b <- d[d$group==1,]
f0 <- lm(y~x+z,data=a);f1 <- lm(y~x+z,data=b)
m0 <- c(1,mean(a$x),mean(a$z));m1 <- c(1,mean(b$x),mean(b$z))
add("gap",mean(b$y)-mean(a$y));add("explained",sum((m1-m0)*coef(f0)))
add("unexplained",sum(m1*(coef(f1)-coef(f0))))
add("explained_x",(mean(b$x)-mean(a$x))*coef(f0)[2])
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

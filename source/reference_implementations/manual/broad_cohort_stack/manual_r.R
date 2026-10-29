# Reference program for broad_cohort_stack (R).
# This is the native R check of the task build. For each cohort, lm() regresses
# y on did and unit and time factors in the balanced comparison sample.
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/broad_cohort_stack.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/broad_cohort_stack"
d <- read.csv(file.path(input_dir, "input.csv"))
rows <- list()
add <- function(term, estimate, se=NA_real_, n=nrow(d)) {
  rows[[length(rows)+1L]] <<- list(term=term,estimate=as.numeric(estimate),se=as.numeric(se),n_obs=as.integer(n))
}
effects <- weights <- numeric(3);total <- 0
for(j in 1:3) {
  cc <- c(3,5,7)[j]
  s <- subset(d,time>=cc-2 & time<=cc+2 & (g==cc | g==0 | g>cc+2))
  s <- s[ave(s$id,s$id,FUN=length)==5,]
  s$did <- as.integer(s$g==cc & s$time>=cc)
  fit <- lm(y~did+factor(id)+factor(time),data=s)
  effects[j] <- coef(fit)["did"];weights[j] <- length(unique(s$id[s$g==cc]));total <- total+nrow(s)
  add(paste0("att_g",cc),effects[j],n=nrow(s))
}
add("att_weighted",weighted.mean(effects,weights),n=total)
cat(jsonlite::toJSON(list(estimates=rows), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

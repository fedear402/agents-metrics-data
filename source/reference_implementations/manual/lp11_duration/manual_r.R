# Native R check for lp11_duration.
# Source: the lp11_duration branch of the six-task native R check
# (six_native.R), which wrote native_r_target.json.
# Route: stacked cohort-month risk sets and survival::clogit with the exact
# conditional likelihood; the first event-set probability by enumeration.
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/lp11_duration.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/lp11_duration"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
library(survival)
d <- read(); rows <- list(); k <- 0
for(g in unique(d$cohort)) for(m in sort(unique(d$end_month[d$cohort==g]))) {
 a <- subset(d,cohort==g & end_month>=m)
 a$exit <- as.integer(a$end_month==m & a$event==1)
 if(sum(a$exit)>0) { k <- k+1; a$riskset <- k; rows[[k]] <- a }
}
a <- do.call(rbind,rows)
fit <- survival::clogit(exit~training+strata(riskset),a,method='exact',control=survival::coxph.control(eps=1e-12))
b <- unname(coef(fit)[1]); first <- rows[[1]]
count <- sum(first$exit); sums <- combn(first$training,count,sum)
logp <- b*sum(first$training*first$exit)-log(sum(exp(b*sums)))
out <- c(training_coefficient=b,conditional_odds_ratio=exp(b),log_likelihood=fit$loglik[2],first_event_set_probability=exp(logp),first_event_set_log_probability=logp)
n <- nrow(d)
result <- list(task_id = "lp11_duration",
               estimates = lapply(names(out), function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = n)))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

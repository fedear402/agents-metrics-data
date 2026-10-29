# Reference program for hard_rolling (R).
# This is the R native check of the task build program (luna-hard-five-seven,
# rolling). It sorts by time and fits each rolling AR(1) with lm on the 99
# pairs inside the 100-observation window.
# Run from the data repository root: Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_rolling.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_rolling"

d <- read.csv(file.path(input_dir, "input.csv")); d <- d[order(d$time),]
pred <- numeric(150)
for(f in 101:250) {
  w <- d$y[(f-100):(f-1)]
  fit <- lm(w[2:100] ~ w[1:99])
  pred[f-100] <- coef(fit)[1]+coef(fit)[2]*w[100]
}
values <- c(sqrt(mean((d$y[101:250]-pred)^2)),pred[c(1,50,100,150)])
out <- data.frame(term=c("rmse","forecast_101","forecast_150","forecast_200","forecast_250"),estimate=values,
                  se=NA_real_,n_obs=c(150L,99L,99L,99L,99L))
cat(jsonlite::toJSON(list(estimates=out), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

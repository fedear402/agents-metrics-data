# Native R check for pilot_probit_interaction_ame.
# Source: the pilot_probit_interaction_ame branch of the six-task native R
# check (six_native.R), which wrote native_r_target.json.
# Route: glm probit with the x*d interaction and x squared, analytic
# derivative for ame_x, and predicted-probability differences for effect_d.
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/pilot_probit_interaction_ame.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/pilot_probit_interaction_ame"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
d <- read(); fit <- glm(y~x*d+z+I(x^2),d,family=binomial(link='probit'),control=glm.control(epsilon=1e-12,maxit=200))
a <- subset(d,target==1); b <- coef(fit)
derivative <- dnorm(predict(fit,a))*(b['x']+b['x:d']*a$d+2*b['I(x^2)']*a$x)
one <- zero <- a; one$d <- 1; zero$d <- 0
out <- c(ame_x=mean(derivative),effect_d=mean(predict(fit,one,type='response')-predict(fit,zero,type='response')))
n <- nrow(d)
result <- list(task_id = "pilot_probit_interaction_ame",
               estimates = lapply(names(out), function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = n)))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

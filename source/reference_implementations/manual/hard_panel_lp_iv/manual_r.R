# Native R check for hard_panel_lp_iv.
# Source: the hard_panel_lp_iv branch of the six-task native R check
# (six_native.R), which wrote native_r_target.json.
# Route: calendar merges for the common origin sample, model.matrix fixed
# effects, and projection 2SLS (qr) at each horizon.
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/hard_panel_lp_iv.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/hard_panel_lp_iv"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
iv <- function(y,C,x,z,w=rep(1,length(y))) {
  X <- cbind(C,x)*sqrt(w); Z <- cbind(C,z)*sqrt(w)
  fitted <- qr.fitted(qr(Z),X)
  as.numeric(qr.coef(qr(fitted),y*sqrt(w)))[ncol(X)]
}
d <- read(); a <- d
for(h in c(-1,1,2)) {
 q <- d[c('unit','period','y')]; q$period <- q$period-h
 names(q)[3] <- paste0('y',h); a <- merge(a,q,by=c('unit','period'))
}
C <- model.matrix(~a[['y-1']]+factor(unit)+factor(period),a)
out <- setNames(sapply(c('y','y1','y2'),function(v) iv(a[[v]],C,a$x,a$z)),paste0('lp_h',0:2))
n <- nrow(a)
result <- list(task_id = "hard_panel_lp_iv",
               estimates = lapply(names(out), function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = n)))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

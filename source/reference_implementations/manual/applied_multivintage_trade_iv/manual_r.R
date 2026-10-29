# Native R check for applied_multivintage_trade_iv.
# Source: the applied_multivintage_trade_iv branch of the six-task native R
# check (six_native.R), which wrote native_r_target.json.
# Route: relational merges, aggregated exposures, and weighted projection 2SLS (qr).
#
# Run from the data repository root:
#   Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/applied_multivintage_trade_iv.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/applied_multivintage_trade_iv"
folder <- input_dir
read <- function(name='input.csv') read.csv(file.path(folder,name))
iv <- function(y,C,x,z,w=rep(1,length(y))) {
  X <- cbind(C,x)*sqrt(w); Z <- cbind(C,z)*sqrt(w)
  fitted <- qr.fitted(qr(Z),X)
  as.numeric(qr.coef(qr(fitted),y*sqrt(w)))[ncol(X)]
}
areas <- read('areas.csv'); emp <- read('employment.csv')
trade <- merge(read('trade.csv'),read('crosswalk.csv'),by='code'); outdata <- read('outcomes.csv')
for(v in c('base','early')) {
 e <- subset(emp,vintage==v)
 totals <- aggregate(employment~period+industry,e,sum); names(totals)[3] <- 'national'
 e <- merge(e,totals,by=c('period','industry'))
 e <- merge(e,subset(areas,vintage==v)[c('area','period','total_employment')],by=c('area','period'))
 e <- merge(e,trade,by=c('period','industry'))
 change <- e[[if(v=='base') 'us_change' else 'other_change']]
 e$value <- e$employment/e$total_employment*change/e$national/(e$price_index/100)/1000*(10/e$years)
 a <- aggregate(value~area+period,e,sum); names(a)[3] <- if(v=='base') 'x' else 'z'
 outdata <- merge(outdata,a,by=c('area','period'))
}
d <- merge(outdata,subset(areas,vintage=='base')[c('area','period','population')],by=c('area','period'))
out <- c(iv_slope=iv(d$change_pp,model.matrix(~period+control,d),d$x,d$z,d$population),exposure_area0_period0=subset(d,area==0 & period==0)$x,exposure_area0_period1=subset(d,area==0 & period==1)$x,instrument_area0_period0=subset(d,area==0 & period==0)$z)
n <- nrow(d)
result <- list(task_id = "applied_multivintage_trade_iv",
               estimates = lapply(names(out), function(k) list(term = k, estimate = unname(out[[k]]), se = NULL, n_obs = n)))
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

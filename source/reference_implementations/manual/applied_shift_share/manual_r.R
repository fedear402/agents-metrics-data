# Reference program for applied_shift_share (R).
# This is the R native check of the task build program (applied-five pilot,
# shift_share). It aggregates the regional residuals to the industry level and
# solves the just-identified weighted IV moment equations with an HC1
# sandwich and the N/(N-2) factor.
# Run from the data repository root: Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/applied_shift_share.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/applied_shift_share"

d <- read.csv(file.path(input_dir, "input.csv"))
r <- d[!duplicated(d$region_id),c("region_id","region_weight","x","y")]
r <- r[order(r$region_id),]
r$e <- r$region_weight/sum(r$region_weight)
zsum <- aggregate(d$share*d$shock,list(region_id=d$region_id),sum)
r$ziv <- zsum$x[match(r$region_id,zsum$region_id)]
for (v in c("x","y","ziv")) r[[paste0(v,"0")]] <- r[[v]]-weighted.mean(r[[v]],r$e)
a <- merge(d[c("region_id","industry_id","share","shock")],r[c("region_id","e","x0","y0","ziv0")],by="region_id")
a$sn <- a$e*a$share
s <- aggregate(cbind(sn=a$sn,xsum=a$sn*a$x0,ysum=a$sn*a$y0,zsum=a$sn*a$ziv0),list(industry_id=a$industry_id),sum)
g <- d[!duplicated(d$industry_id),c("industry_id","shock")]
s$shock <- g$shock[match(s$industry_id,g$industry_id)]
s$xbar <- s$xsum/s$sn
s$ybar <- s$ysum/s$sn
s$zbar <- s$zsum/s$sn

# Just-identified weighted IV through the two sample moment equations.
iv_hc1 <- function(y,x,g,w) {
  X <- cbind(1,x)
  Z <- cbind(1,g)
  C <- crossprod(Z,w*X)
  b <- solve(C,crossprod(Z,w*y))
  u <- drop(y-X%*%b)
  scores <- Z*(w*u)
  invC <- solve(C)
  V <- invC%*%crossprod(scores)%*%t(invC)*length(y)/(length(y)-2)
  c(estimate=b[2],se=sqrt(V[2,2]))
}
eff <- iv_hc1(s$ybar,s$xbar,s$shock,s$sn)
fst <- iv_hc1(s$xbar,s$zbar,s$shock,s$sn)
region_eff <- iv_hc1(r$y,r$x,r$ziv,r$e)
region_fst <- coef(lm(x~ziv,data=r,weights=e))["ziv"]
stopifnot(abs(eff["estimate"]-region_eff["estimate"])<1e-10)
stopifnot(abs(fst["estimate"]-region_fst)<1e-10)
out <- data.frame(term=c("effect","first_stage","effective_shocks"),estimate=c(eff["estimate"],fst["estimate"],1/sum(s$sn^2)),se=c(eff["se"],fst["se"],NA),n_obs=nrow(s))
rownames(out) <- NULL
cat(jsonlite::toJSON(list(estimates=out), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

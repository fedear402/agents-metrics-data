# Reference program for applied_cumulative_hac (R).
# This is the R native check of the task build program (applied-five pilot,
# cumulative_hac). It uses lm.fit and builds the Bartlett lag cross-products
# with 15 lags by hand, then multiplies the covariance by n/(n-16).
# Run from the data repository root: Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/applied_cumulative_hac.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/applied_cumulative_hac"

d<-read.csv(file.path(input_dir, "input.csv")); d<-d[order(d$date),]
change<-c(NA,diff(log(d$vix)))
lagged<-sapply(1:15,function(j)c(rep(NA,j),head(change,-j)))
ok<-complete.cases(lagged); X<-cbind(1,lagged[ok,]); y<-log(d$sales[ok])
fit<-lm.fit(X,y); b<-fit$coefficients; U<-X*fit$residuals; n<-nrow(X)
M<-crossprod(U)
for(j in 1:15) {
 G<-crossprod(U[(j+1):n,,drop=FALSE],U[1:(n-j),,drop=FALSE])
 M<-M+(1-j/16)*(G+t(G))
}
A<-solve(crossprod(X)); V<-A%*%M%*%A*n/(n-16)
C<-matrix(0,3,16); C[1,2:16]<-1; C[2,2]<-1; C[3,16]<-1
out<-data.frame(term=c("cumulative_response","lag_1","lag_15"),estimate=drop(C%*%b),se=sqrt(diag(C%*%V%*%t(C))),n_obs=n)
cat(jsonlite::toJSON(list(estimates=out), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

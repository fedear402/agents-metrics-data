# Reference program for applied_fuzzy_rd (R).
# This is the R native check of the task build program (applied-five pilot,
# fuzzy_rd). It fits weighted lm with separate slopes for earnings and
# participation, builds HC0 influence functions of the two jumps, and uses
# the delta method with their joint covariance for the ratio.
# Run from the data repository root: Rscript manual_r.R [INPUT_DIR]
# INPUT_DIR defaults to tasks/applied_fuzzy_rd.
# The program prints one JSON object and writes no files.
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else "tasks/applied_fuzzy_rd"

d<-read.csv(file.path(input_dir, "input.csv")); d<-subset(d,abs(score)<1)
d$right<-as.numeric(d$score>=0); d$w<-1-abs(d$score)
fy<-lm(earnings~right*score,data=d,weights=w)
fd<-lm(participated~right*score,data=d,weights=w)
X<-model.matrix(fy); A<-solve(crossprod(X,d$w*X))
ay<-drop(X%*%A[,2])*d$w*residuals(fy)
ad<-drop(X%*%A[,2])*d$w*residuals(fd)
V<-crossprod(cbind(ay,ad)); rf<-coef(fy)[2]; fs<-coef(fd)[2]
g<-c(1/fs,-rf/fs^2)
out<-data.frame(term=c("treatment_effect","reduced_form","first_stage"),estimate=c(rf/fs,rf,fs),se=c(sqrt(drop(t(g)%*%V%*%g)),sqrt(diag(V))),n_obs=nrow(d))
rownames(out)<-NULL
cat(jsonlite::toJSON(list(estimates=out), auto_unbox = TRUE, digits = NA, null = "null", na = "null"), "\n")

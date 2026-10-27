"""Package three fixed nonlinear tasks and verify native reference calculations.

Run with uv run python code/benchmark/task_definitions/nonlinear_production.py.
This program makes no provider calls and preserves the original task inputs.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'quality_reports/reports/2026-09-17_twenty-harder-tasks/groups/nonlinear'
OUT = ROOT / 'data/tasks/paper22_2026-09-18'
TASKS = ('hard_heteroskedastic_probit', 'hard_truncated_normal_effects', 'hard_poisson_hurdle_effects')
STATA = '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se'
SOURCE_IDS = ('neumark','badunenko','pohlmeier')
SOURCE_NOTES = [
    'Neumark and Rich (2019), pages 236–238, use heteroskedastic probit for correspondence experiments. Their identifying restriction equates selected qualification effects across groups. Their decomposition includes level and variance channels. This task uses simulated data, a normalized scale equation, a continuous derivative, and an average binary probability contrast. The binary variable enters only the mean equation. It does not reproduce the paper estimate or derivative decomposition.',
    'Badunenko and Tauchmann (2018), Sections 2 and 4, apply truncated-normal regression to production inefficiency scores. Their outcome uses data envelopment analysis and has support beginning at one. Their full procedure requires bootstrap inference for estimated scores. This task uses directly simulated observations sampled above zero. It retains the conditional normal likelihood and conditional-mean effects. It does not reproduce frontier estimation, generated scores, or bootstrap inference.',
    'Pohlmeier and Ulrich (1992), pages 6–9 and 13, separate physician contact and positive visit frequency. Their application uses negative-binomial-I probabilities and compares nested Poisson hurdle specifications. This task uses simulated counts with logit contact probability and zero-truncated Poisson positive counts. Its unconditional mean, continuous derivative, binary contrast, and upper-tail probability follow that stated synthetic specification. It does not reproduce the paper distribution choices or empirical estimates.',
]

R_EXAMPLES = [
'''# Known coefficients in a smaller mean/scale model: mu=0.3+0.8*x, sigma=exp(0.5*z).
x <- c(-1,0,1); z <- c(0,1,-1)
sigma <- exp(0.5*z); eta <- (0.3+0.8*x)/sigma
cat('probabilities', paste(sprintf('%.10f', pnorm(eta)), collapse=' '), '\\n')
cat('ame_x', sprintf('%.10f', mean(dnorm(eta)*0.8/sigma)), '\\n')
''',
'''# A simpler intercept-and-x truncated model has known beta=(0.2,0.6), sigma=1.
x <- c(-0.5,0.5,1); mu <- 0.2+0.6*x; a <- -mu
lam <- dnorm(a)/pnorm(a, lower.tail=FALSE)
conditional_mean <- mu+lam
derivative <- 0.6*(1+a*lam-lam^2)
cat('mean_observed', sprintf('%.10f', mean(conditional_mean)), '\\n')
cat('ame_x', sprintf('%.10f', mean(derivative)), '\\n')
''',
'''# A simpler known-parameter hurdle model on two covariate rows.
x <- c(-0.5,0.5); p <- plogis(-0.2+0.4*x); lam <- exp(0.1+0.3*x)
q <- -expm1(-lam); conditional <- lam/q
mu <- p*conditional
derivative <- p*(1-p)*0.4*conditional+p*0.3*lam*(q-lam*exp(-lam))/q^2
cat('mean_y', sprintf('%.10f', mean(mu)), '\\n')
cat('ame_x', sprintf('%.10f', mean(derivative)), '\\n')
cat('prob_at_least_three', sprintf('%.10f', mean(p*ppois(2,lam,lower.tail=FALSE)/q)), '\\n')
''']

STATA_EXAMPLES = [
'''clear
input double x z
-1 0
0 1
1 -1
end
* Known coefficients: mu=0.3+0.8*x, sigma=exp(0.5*z).
generate double sigma = exp(0.5*z)
generate double eta = (0.3+0.8*x)/sigma
generate double probability = normal(eta)
display "probabilities " %15.10f probability[1] " " %15.10f probability[2] " " %15.10f probability[3]
generate double derivative = normalden(eta)*0.8/sigma
quietly summarize derivative
display "ame_x " %15.10f r(mean)
''',
'''clear
input double x
-0.5
0.5
1
end
* Known beta=(0.2,0.6), sigma=1 in a simpler truncated model.
generate double mu = 0.2+0.6*x
generate double a = -mu
generate double lam = normalden(a)/normal(-a)
generate double conditional_mean = mu+lam
generate double derivative = 0.6*(1+a*lam-lam^2)
quietly summarize conditional_mean
display "mean_observed " %15.10f r(mean)
quietly summarize derivative
display "ame_x " %15.10f r(mean)
''',
'''clear
input double x
-0.5
0.5
end
* A simpler known-parameter hurdle model on two covariate rows.
generate double p = invlogit(-0.2+0.4*x)
generate double lam = exp(0.1+0.3*x)
generate double q = 1-exp(-lam)
generate double conditional = lam/q
generate double mu = p*conditional
generate double derivative = p*(1-p)*0.4*conditional+p*0.3*lam*(q-lam*exp(-lam))/q^2
generate double tail = p*poissontail(lam,3)/q
quietly summarize mu
display "mean_y " %15.10f r(mean)
quietly summarize derivative
display "ame_x " %15.10f r(mean)
quietly summarize tail
display "prob_at_least_three " %15.10f r(mean)
''']

R_REFERENCES = [
'''X <- cbind(1,d$x,d$d); sign <- 2*d$y-1
objective <- function(b) -sum(pnorm(sign*as.vector(X%*%b[1:3])*exp(-b[4]*d$z),log.p=TRUE))
gradient <- function(b) {
 inv <- exp(-b[4]*d$z); eta <- as.vector(X%*%b[1:3])*inv; t <- sign*eta
 r <- sign*exp(dnorm(t,log=TRUE)-pnorm(t,log.p=TRUE))
 -c(crossprod(X,r*inv),sum(r*(-eta*d$z)))
}
fit <- optim(c(0,.4,-.3,.2),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0, max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; sigma <- exp(b[4]*d$z); mu <- as.vector(X%*%b[1:3])
out <- c(gamma_z=b[4],ame_x=mean(dnorm(mu/sigma)*b[2]/sigma),effect_d=mean(pnorm((b[1]+b[2]*d$x+b[3])/sigma)-pnorm((b[1]+b[2]*d$x)/sigma)))
''',
'''X <- cbind(1,d$x,d$z)
objective <- function(b) {
 sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3])
 -sum(dnorm((d$y-mu)/sigma,log=TRUE)-b[4]-pnorm(mu/sigma,log.p=TRUE))
}
gradient <- function(b) {
 sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3]); a <- -mu/sigma; t <- (d$y-mu)/sigma
 lam <- exp(dnorm(a,log=TRUE)-pnorm(-a,log.p=TRUE))
 -c(crossprod(X,(t-lam)/sigma),sum(t*t-1-a*lam))
}
fit <- optim(c(0,.5,0,0),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0,max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; sigma <- exp(b[4]); mu <- as.vector(X%*%b[1:3]); a <- -mu/sigma
lam <- exp(dnorm(a,log=TRUE)-pnorm(-a,log.p=TRUE))
out <- c(mean_observed=mean(mu+sigma*lam),ame_x=mean(b[2]*(1+a*lam-lam^2)),latent_x=b[2])
''',
'''X <- cbind(1,d$x,d$d); positive <- d$y>0; XP <- X[positive,,drop=FALSE]; y <- d$y[positive]
fit_logit <- glm(positive~x+d,data=d,family=binomial(),control=glm.control(epsilon=1e-12,maxit=200))
stopifnot(fit_logit$converged)
a <- unname(coef(fit_logit))
objective <- function(b) {
 eta <- as.vector(XP%*%b); lam <- exp(eta)
 -sum(y*eta-lam-lgamma(y+1)-log(-expm1(-lam)))
}
gradient <- function(b) {lam <- exp(as.vector(XP%*%b)); -as.vector(crossprod(XP,y-lam/(-expm1(-lam))))}
fit <- optim(c(0,.2,.2),objective,gradient,method='BFGS',control=list(reltol=1e-13,maxit=5000))
stopifnot(fit$convergence==0,max(abs(gradient(fit$par)))<1e-3)
b <- fit$par; p <- plogis(as.vector(X%*%a)); lam <- exp(as.vector(X%*%b)); q <- -expm1(-lam)
mean_fun <- function(xx) {l <- exp(as.vector(xx%*%b)); plogis(as.vector(xx%*%a))*l/(-expm1(-l))}
one <- zero <- X; one[,3] <- 1; zero[,3] <- 0
out <- c(mean_y=mean(p*lam/q),ame_x=mean(p*(1-p)*a[2]*lam/q+p*b[2]*lam*(q-lam*exp(-lam))/q^2),effect_d=mean(mean_fun(one)-mean_fun(zero)),prob_at_least_three=mean(p*ppois(2,lam,lower.tail=FALSE)/q))
''']

STATA_REFERENCES = [
'''hetprobit y c.x i.d, het(z) nolog
display "CHECK gamma_z " %21.12f _b[lnsigma:z]
margins, dydx(x d) predict(pr)
matrix M = r(b)
display "CHECK ame_x " %21.12f M[1,1]
display "CHECK effect_d " %21.12f M[1,colnumb(M,"1.d")]
''',
'''truncreg y x z, ll(0) nolog
display "CHECK latent_x " %21.12f _b[x]
predict double cm, e(0,.)
quietly summarize cm
display "CHECK mean_observed " %21.12f r(mean)
margins, dydx(x) predict(e(0,.))
matrix M = r(b)
display "CHECK ame_x " %21.12f M[1,1]
''',
'''generate byte positive = y>0
logit positive x d, nolog
predict double eta, xb
scalar a_x = _b[x]
scalar a_d = _b[d]
tpoisson y x d if positive, ll(0) nolog
predict double loglambda, xb
scalar b_x = _b[x]
scalar b_d = _b[d]
generate double p = invlogit(eta)
generate double lambda = exp(loglambda)
generate double q = 1-exp(-lambda)
generate double mu = p*lambda/q
generate double deriv = p*(1-p)*a_x*lambda/q + p*b_x*lambda*(q-lambda*exp(-lambda))/(q*q)
generate double lambda1 = exp(loglambda+b_d*(1-d))
generate double lambda0 = exp(loglambda-b_d*d)
generate double delta = invlogit(eta+a_d*(1-d))*lambda1/(1-exp(-lambda1))-invlogit(eta-a_d*d)*lambda0/(1-exp(-lambda0))
generate double tail = p*poissontail(lambda,3)/q
quietly summarize mu
display "CHECK mean_y " %21.12f r(mean)
quietly summarize deriv
display "CHECK ame_x " %21.12f r(mean)
quietly summarize delta
display "CHECK effect_d " %21.12f r(mean)
quietly summarize tail
display "CHECK prob_at_least_three " %21.12f r(mean)
''']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, cwd, label):
    result = subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=180)
    (cwd/(label+'.stdout')).write_text(result.stdout)
    (cwd/(label+'.stderr')).write_text(result.stderr)
    assert result.returncode == 0, (argv,result.stderr)
    return result.stdout


def stata_run(script, work):
    """Execute a native check while main holds the shared Stata lock."""
    run([STATA,'-b','do',str(script)],work,script.stem)
    text = (work/(script.stem+'.log')).read_text()
    assert 'VALIDATION_FINISHED' in text and not re.search(r'^r\(\d+\);',text,re.M), script
    return text


def example_numbers(text):
    numbers=[]
    for line in text.splitlines():
        if re.match(r'^(probabilities|ame_x|mean_observed|mean_y|prob_at_least_three)\s',line):
            numbers.extend(float(x) for x in re.findall(r'-?\d+\.\d+(?:e[+-]?\d+)?',line))
    return numbers


def finalize_packages():
    """Publish task readiness only after saved native checks are complete."""
    sources=json.loads((ROOT/'quality_reports/reports/2026-09-17_applied-next-eight/SOURCES.json').read_text())
    for i,task in enumerate(TASKS):
        folder=OUT/task
        validation=json.loads((folder/'work/native_validation.json').read_text())
        assert validation['passed'] and all(row['passed'] for row in validation['checks'])
        assert validation['python_zero_shot_unchanged'] and validation['python_few_shot_unchanged']
        spec=json.loads((folder/'task.json').read_text())
        source=next(row for row in sources if row['id']==SOURCE_IDS[i])
        assert sha(ROOT/source['local_path'])==source['sha256']
        save(folder/'source.json',source)
        (folder/'README.md').write_text('# '+spec['title']+'\n\n'+SOURCE_NOTES[i]+'\n\nThe statement, Python example, Python prompt fragments, reference, and original input path are preserved. R and Stata use translations of the same single example. No binary contrast or estimation routine was added to an example.\n\nThe source PDF hash matches the downloaded source record. The source review is in quality_reports/reports/2026-09-17_applied-next-eight/literature_review.md. Native R and Stata references and all language examples passed. Numerical checks and script hashes are in work/native_validation.json.\n\nRegenerate and verify with `uv run python code/benchmark/task_definitions/nonlinear_production.py`. Stata checks use the existing native execution lock. Raw input data stay at the path in task.json.\n')
        save(folder/'ready.json',dict(passed=True,reference_languages=['python','r','stata'],example_languages=['python','r','stata'],validation='work/native_validation.json',raw_data_copied=False))


def main():
    catalog=json.loads((SOURCE/'collector_catalog.json').read_text())
    for task in TASKS:
        (OUT/task/'ready.json').unlink(missing_ok=True)
    for i,task in enumerate(TASKS):
        spec=catalog[task]; folder=OUT/task; work=folder/'work'; fragments=folder/'fragments'
        work.mkdir(parents=True,exist_ok=True); fragments.mkdir(exist_ok=True)
        original=SOURCE/task
        (folder/'statement.txt').write_text(spec['prompt'])
        for name in ('example.py','reference.json'):
            (folder/name).write_bytes((original/name).read_bytes())
        (folder/'example.R').write_text(R_EXAMPLES[i])
        (folder/'example.do').write_text(STATA_EXAMPLES[i])
        example_outputs={}
        example_outputs['python']=run([sys.executable,str(folder/'example.py')],work,'example_python')
        example_outputs['r']=run(['Rscript','--vanilla',str(folder/'example.R')],work,'example_r')
        wrapper=work/'example_stata.do'
        wrapper.write_text('version 19\nset more off\n'+STATA_EXAMPLES[i]+'display "VALIDATION_FINISHED"\nexit, clear\n')
        example_outputs['stata']=stata_run(wrapper,work)
        expected=example_numbers(example_outputs['python'])
        assert expected
        for language in ('r','stata'):
            got=example_numbers(example_outputs[language])
            assert len(got)==len(expected) and all(abs(a-b)<1e-9 for a,b in zip(got,expected)),(task,language,got,expected)
        for condition in ('zero_shot','few_shot'):
            text=(original/(condition+'.txt')).read_text()
            (fragments/('python_'+condition+'.txt')).write_text(text)
            for language,suffix,label in [('r','R','R'),('stata','do','Stata')]:
                translated=text.replace('self-contained Python script','self-contained '+label+' script')
                if condition=='few_shot':
                    example=(folder/('example.'+suffix)).read_text().rstrip()
                    translated=re.sub(r'```python\n.*?\n```',lambda match:'```'+language+'\n'+example+'\n```',translated,count=1,flags=re.S)
                if language=='r':
                    translated=translated.replace('with json.dump','with jsonlite::write_json').replace('"platform": "python"','"platform": "r"')
                else:
                    translated=translated.split('\nReporting:\n')[0]+'\nReporting:\nReport point estimates only. Print one line for each term. Use NA for se and the full row count for n_obs. Report at least eight significant digits.\n'+''.join('ANSWER: '+term+' estimate=<number> se=NA n_obs=<integer>\n' for term in spec['estimands'])
                assert spec['prompt'] in translated
                (fragments/(language+'_'+condition+'.txt')).write_text(translated)
        data=Path(spec['data_path'])
        reference_r=folder/'reference.R'
        reference_r.write_text('d <- read.csv('+json.dumps(str(data))+')\n'+R_REFERENCES[i]+"for (term in names(out)) cat('CHECK',term,sprintf('%.12f',out[[term]]),'\\n')\n")
        reference_stata=folder/'reference.do'
        reference_stata.write_text('version 19\nclear all\nset more off\nimport delimited '+json.dumps(str(data))+', clear\n'+STATA_REFERENCES[i]+'display "VALIDATION_FINISHED"\nexit, clear\n')
        outputs={'r':run(['Rscript','--vanilla',str(reference_r)],work,'reference_r'),'stata':stata_run(reference_stata,work)}
        checks=[]
        for language,text in outputs.items():
            values={term:float(value) for term,value in re.findall(r'^CHECK\s+(\w+)\s+([-+\d.eE]+)',text,re.M)}
            assert set(values)==set(spec['estimands']),(task,language,values)
            for row in spec['reference']['estimates']:
                error=abs(values[row['term']]-row['estimate'])
                checks.append(dict(language=language,term=row['term'],value=values[row['term']],reference=row['estimate'],absolute_error=error,passed=error<1e-5))
        assert all(row['passed'] for row in checks),checks
        save(folder/'task.json',spec)
        save(work/'native_validation.json',dict(passed=True,checks=checks,examples_passed=['python','r','stata'],source_input_sha256=sha(data),python_zero_shot_unchanged=sha(fragments/'python_zero_shot.txt')==sha(original/'zero_shot.txt'),python_few_shot_unchanged=sha(fragments/'python_few_shot.txt')==sha(original/'few_shot.txt'),scripts={str(path.relative_to(ROOT)):sha(path) for path in (reference_r,reference_stata,folder/'example.py',folder/'example.R',folder/'example.do')}))
        stamp=datetime.now().strftime('%H:%M')
        (work/'journal.md').write_text(f'## {stamp} - Native task package\nPreserved the Python task, example, reference, and original input path.\nExecuted native R and Stata references and all three language examples.\nAll requested estimates match the reference within an absolute difference of 0.00001.\nNo provider calls or historical record changes occurred.\n')
        save(work/'checkpoint.json',dict(task=task,status='native_validation_complete',provider_calls=0))
        print(task,'PASS',len(checks),'reference checks',flush=True)
    finalize_packages()


if __name__=='__main__':
    with open('/private/tmp/agents_metrics_native_stata.lock','a') as lock:
        print('Waiting for the shared Stata lock for all three task checks.',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        print('Acquired the shared Stata lock.',flush=True)
        main()

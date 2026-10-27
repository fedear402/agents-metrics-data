"""Run labels describe task success: scored estimates and, in nine tasks, scored standard errors."""

import json
import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "falstuff_site/app.js"


def test_pass_label_does_not_claim_all_diagnostics_passed():
    script = r"""
const fs = require('fs');
const source = fs.readFileSync(process.argv[1], 'utf8');
const start = source.indexOf('  function comboOutcomeText(');
const end = source.indexOf('\n  function ', start + 1);
const fn = source.slice(start, end);
eval(fn);
console.log(JSON.stringify([
  comboOutcomeText({pass:true,failed:[],terms:[{term:'x',pass:true}]}),
  comboOutcomeText({pass:true,failed:[],terms:[{term:'x',pass:true,seScored:true,sePass:true}]}),
  comboOutcomeText({pass:false,failed:['Silent failure'],terms:[{term:'x',pass:true,seScored:true,sePass:false}]}),
  comboOutcomeText({pass:false,failed:['Loud failure'],terms:[]}),
]));
"""
    result = subprocess.run(
        ["node", "-e", script, str(APP)], check=True, capture_output=True, text=True
    )
    estimates_only, with_se, se_failure, loud = json.loads(result.stdout)
    assert estimates_only == "Scored estimates agree with the reference output"
    assert with_se == "Scored estimates and standard errors agree with the reference output"
    # A run that fails only on its scored standard error names it.
    assert se_failure == "Silent failure. Standard error outside tolerance or missing: x."
    assert loud == "Loud failure. No complete output to compare with the reference output."

"""The rendered run follows the interaction order and escapes saved text."""

import json
import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "falstuff_site/app.js"


def test_run_flow_order_full_result_and_text_escaping():
    script = r"""
const fs = require('fs');
const source = fs.readFileSync(process.argv[1], 'utf8');
const start = source.indexOf('  function renderRunText(');
const end = source.indexOf('  function renderComboView(', start);
function escapeHtml(s) { return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }
function comboFactsLine() { return ''; }
function statusClass() { return 'pass'; }
function comboOutcomeText() { return 'Reference agreement and submission validity passed'; }
function renderComboTerms() { return '<table></table>'; }
eval(source.slice(start,end));
const combo={pass:true,result:'old-excerpt',flow:{prompt:{status:'saved',text:'<script>alert(1)</script>'},process:{events:[{label:'First command',text:'step1'},{label:'Second command',text:'step2'}]},final:{text:'final-answer',script:'submitted-code'},resultText:'full-result',executionText:'last-execution'}};
console.log(JSON.stringify({html:renderRunFlow(combo,[]),missing:renderRunFlow({pass:false},[])}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(APP)], check=True, capture_output=True, text=True
    )
    output = json.loads(result.stdout)
    html = output["html"]
    positions = [
        html.index(x)
        for x in ["Prompt sent", "Intermediate steps", "Final answer", ">Result<"]
    ]
    assert positions == sorted(positions)
    assert html.index("First command") < html.index("Second command")
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "final-answer" in html and "submitted-code" in html
    assert "full-result" in html and "old-excerpt" not in html
    assert "last-execution" in html
    assert "exact initial prompt is not available" in output["missing"]
    assert (
        "No intermediate messages or tool activity were retained" in output["missing"]
    )

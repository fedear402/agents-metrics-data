"""Regression tests for cost-basis labels in the public result detail."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = PROJECT_ROOT / "code" / "falstuff_site" / "app.js"


def test_result_cost_labels_state_the_cost_basis() -> None:
    script = r"""
const fs = require("fs");
const vm = require("vm");

const appPath = process.argv[1];
let source = fs.readFileSync(appPath, "utf8");
const startup = '  document.addEventListener("DOMContentLoaded", init);\n})();';
if (!source.includes(startup)) {
  throw new Error("Could not isolate the app helpers from startup");
}
source = source.replace(
  startup,
  "  globalThis.__renderResultDetailForTest = renderResultDetail;\n})();",
);

const resultDetail = {
  innerHTML: "",
  querySelector: () => null,
};
const sandbox = {
  window: {
    AGENTS_METRICS_SITE_DATA: {},
    AGENTS_METRICS_RUN_TRACES: {},
  },
  document: {
    getElementById: (id) => {
      if (id !== "resultDetail") {
        throw new Error(`Unexpected element request: ${id}`);
      }
      return resultDetail;
    },
  },
};
vm.runInNewContext(source, sandbox, { filename: appPath });

const render = (costBasis) => {
  sandbox.__renderResultDetailForTest({
    label: "Test row",
    records: 5,
    taskSuccessPassed: 4,
    success: 80,
    criteria: [],
    cost: 1,
    meanCost: 0.2,
    costPerSuccess: 0.25,
    costBasis,
  }, "models");
  return resultDetail.innerHTML;
};
process.stdout.write(JSON.stringify({
  list_price: render("list_price"),
  billed: render("billed"),
  imputed: render("imputed"),
  missing: render(undefined),
  unexpected: render("vendor-reported"),
}));
"""
    completed = subprocess.run(
        ["node", "--eval", script, str(APP_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = json.loads(completed.stdout)
    expected = {
        # The site bundle prices every run at API list prices (build_site_data.py).
        "list_price": (
            "List-price total cost",
            "List-price mean cost per run",
            "List-price cost per successful run",
        ),
        "billed": (
            "Billed total cost",
            "Billed mean cost per run",
            "Billed cost per success",
        ),
        "imputed": (
            "List-price total estimate",
            "List-price estimate per run",
            "List-price estimate per success",
        ),
        "missing": (
            "Total cost (basis not stated)",
            "Mean cost per run (basis not stated)",
            "Cost per success (basis not stated)",
        ),
        "unexpected": (
            "Total cost (basis not stated)",
            "Mean cost per run (basis not stated)",
            "Cost per success (basis not stated)",
        ),
    }
    for basis, labels in expected.items():
        for label in labels:
            assert f"<span>{label}</span>" in rendered[basis]

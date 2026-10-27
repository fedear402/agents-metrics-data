"""Regression test for task naming in the public explorer.

Tasks carry no number on any display surface. The kicker beside a task name is
its group, which arrives in the generated bundle from
``code/benchmark/task_display.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = PROJECT_ROOT / "code" / "falstuff_site" / "app.js"

TASK_NUMBER_RE = re.compile(r"\bT\d{2}\b")


def test_task_kicker_is_the_group_and_carries_no_task_number() -> None:
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
  [
    "  globalThis.__renderedTasksForTest = tasks.map((item) => ({",
    "    id: item.id,",
    "    group: taskGroup(item),",
    "    title: displayTaskTitle(item),",
    "  }));",
    "})();",
  ].join("\n"),
);

const rendered = (siteData) => {
  const sandbox = {
    window: {
      AGENTS_METRICS_SITE_DATA: siteData,
      AGENTS_METRICS_RUN_TRACES: {},
    },
  };
  vm.runInNewContext(source, sandbox, { filename: appPath });
  return sandbox.__renderedTasksForTest;
};
process.stdout.write(JSON.stringify({
  generated: rendered({
    tasks: [
      { id: "probit", title: "Probit regression", group: "Binary regression" },
      { id: "iv2sls_robust", title: "Two-stage least squares", group: "Instrumental variables" },
      { id: "cox_hazard", title: "Cox regression", group: "Other" },
    ],
  }),
  withoutGroup: rendered({ tasks: [{ id: "probit", title: "Probit regression" }] }),
  withoutData: rendered({}),
}));
"""
    completed = subprocess.run(
        ["node", "--eval", script, str(APP_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["generated"] == [
        {
            "id": "probit",
            "group": "Binary regression",
            "title": "Probit regression",
        },
        {
            "id": "iv2sls_robust",
            "group": "Instrumental variables",
            "title": "Two-stage least squares",
        },
        {"id": "cox_hazard", "group": "Other", "title": "Cox regression"},
    ]
    # A bundle without a group still renders, and never invents a number.
    assert result["withoutGroup"] == [
        {"id": "probit", "group": "Other", "title": "Probit regression"}
    ]
    # The client keeps no task list of its own. Without the generated bundle it
    # renders nothing rather than a stale set of names and rates.
    assert result["withoutData"] == []


def test_client_never_prints_a_task_number() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "function taskNumber(" not in source
    assert TASK_NUMBER_RE.search(source) is None

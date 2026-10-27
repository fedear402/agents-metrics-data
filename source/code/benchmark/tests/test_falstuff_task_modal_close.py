"""Regression tests for task-dialog close behavior in the public explorer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = PROJECT_ROOT / "code" / "falstuff_site" / "app.js"


def test_task_modal_close_paths_leave_the_dialog_closed() -> None:
    """Escape, pointer close, and hash navigation must keep the dialog closed."""
    script = r"""
const fs = require("fs");
const vm = require("vm");

const appPath = process.argv[1];
let source = fs.readFileSync(appPath, "utf8");
const startup = '  document.addEventListener("DOMContentLoaded", init);\n})();';
if (!source.includes(startup)) {
  throw new Error("Could not isolate the task-modal helpers from startup");
}
source = source.replace(
  startup,
  [
    "  globalThis.__taskModalForTest = { setupTaskModal, closeTaskModal, openTaskModal };",
    "})();",
  ].join("\n"),
);
source = source.replace(
  "    renderTaskDetail(item, options);\n",
  "    if (options.updateHash) setTaskHash(item.id);\n",
);

class ClassList {
  constructor(initial = []) {
    this.values = new Set(initial);
  }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  contains(value) { return this.values.has(value); }
  toggle(value, force) {
    if (force) this.add(value);
    else this.remove(value);
  }
}

class Button {
  constructor() { this.listeners = {}; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  click() { this.listeners.click({ target: this }); }
}

class NativeDialog {
  constructor(closeButton) {
    this.open = true;
    this.closeButton = closeButton;
    this.restoreButton = null;
    this.listeners = {};
  }
  querySelector(selector) {
    if (selector === ".modal-close") return this.closeButton;
    throw new Error(`Unexpected dialog selector: ${selector}`);
  }
  addEventListener(type, listener) {
    this.listeners[type] = [...(this.listeners[type] || []), listener];
  }
  dispatch(type, event = { target: this }) {
    for (const listener of this.listeners[type] || []) listener(event);
  }
  close() {
    this.open = false;
    this.dispatch("close");
  }
  showModal() { this.open = true; }
  requestEscapeClose() {
    const event = {
      target: this,
      defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; },
    };
    this.dispatch("cancel", event);
    if (!event.defaultPrevented && this.open) {
      // Chrome removes the open state, restores focus, and emits a synthetic
      // click before it sends the delayed close event.
      this.open = false;
      this.restoreButton?.click();
      this.dispatch("close");
    }
  }
}

const closeButton = new Button();
const openButton = new Button();
const modal = new NativeDialog(closeButton);
modal.restoreButton = openButton;
const bodyClasses = new ClassList(["modal-open"]);
const selectedRow = {
  dataset: { taskRow: "staggered_did" },
  classList: new ClassList(["is-selected"]),
};
const location = {
  pathname: "/chatbots-agents-econometrics/explore.html",
  search: "",
  hash: "#task=staggered_did",
};
let replaceCalls = 0;
const sandbox = {
  window: {
    AGENTS_METRICS_SITE_DATA: {},
    AGENTS_METRICS_RUN_TRACES: {},
    location,
    history: {
      replaceState: (_state, _title, url) => {
        replaceCalls += 1;
        location.hash = url.includes("#") ? url.slice(url.indexOf("#")) : "";
      },
    },
    setTimeout,
  },
  document: {
    body: { classList: bodyClasses },
    getElementById: (id) => {
      if (id === "taskModal") return modal;
      return null;
    },
    querySelectorAll: (selector) => {
      if (selector === "[data-task-row]") return [selectedRow];
      throw new Error(`Unexpected document selector: ${selector}`);
    },
  },
};
vm.runInNewContext(source, sandbox, { filename: appPath });
sandbox.__taskModalForTest.setupTaskModal();
openButton.addEventListener("click", () => {
  sandbox.__taskModalForTest.openTaskModal(
    { id: "staggered_did" },
    { updateHash: true },
  );
});

modal.requestEscapeClose();
const escape = {
  open: modal.open,
  hash: location.hash,
  bodyModalOpen: bodyClasses.contains("modal-open"),
  selected: selectedRow.classList.contains("is-selected"),
  replaceCalls,
};

modal.open = true;
location.hash = "#task=staggered_did";
bodyClasses.add("modal-open");
selectedRow.classList.add("is-selected");
closeButton.click();
const pointer = {
  open: modal.open,
  hash: location.hash,
  bodyModalOpen: bodyClasses.contains("modal-open"),
  selected: selectedRow.classList.contains("is-selected"),
};

modal.open = true;
location.hash = "";
bodyClasses.add("modal-open");
selectedRow.classList.add("is-selected");
sandbox.__taskModalForTest.closeTaskModal({ clearHash: false });
const hashNavigation = {
  open: modal.open,
  hash: location.hash,
  bodyModalOpen: bodyClasses.contains("modal-open"),
  selected: selectedRow.classList.contains("is-selected"),
};

setTimeout(() => {
  modal.open = false;
  location.hash = "";
  bodyClasses.remove("modal-open");
  openButton.click();
  const nextTurnOpen = {
    open: modal.open,
    hash: location.hash,
    bodyModalOpen: bodyClasses.contains("modal-open"),
  };
  process.stdout.write(JSON.stringify({ escape, pointer, hashNavigation, nextTurnOpen }));
}, 0);
"""
    completed = subprocess.run(
        ["node", "--eval", script, str(APP_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    expected = {
        "open": False,
        "hash": "",
        "bodyModalOpen": False,
        "selected": False,
    }
    assert result["escape"] == {**expected, "replaceCalls": 1}
    assert result["pointer"] == expected
    assert result["hashNavigation"] == expected
    assert result["nextTurnOpen"] == {
        "open": True,
        "hash": "#task=staggered_did",
        "bodyModalOpen": True,
    }

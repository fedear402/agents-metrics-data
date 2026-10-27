"""The shared raw-data inventory detects incomplete or changed checkouts."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_data", Path(__file__).parents[1] / "verify_data.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tree(root):
    for directory in module.ROOTS:
        (root / directory).mkdir(parents=True)
    (root / "data/runs/response.txt").write_bytes(b"original response\r\n")
    (root / "data/runs/python").symlink_to("/original/machine/python")
    (root / "data/analysis/counts.json").write_text('{"n":1}')
    (root / "data/tasks/input.csv").write_text("x\n1\n")


def test_verifies_every_byte_and_preserves_link_target(tmp_path):
    tree(tmp_path)
    inventory = tmp_path / "inventory.gz"
    module.write_inventory(tmp_path, inventory)
    result = module.verify(tmp_path, inventory)
    assert result["verified_files"] == 3
    assert result["verified_links"] == 1


@pytest.mark.parametrize("change", ["add", "edit", "remove", "link"])
def test_detects_source_drift(tmp_path, change):
    tree(tmp_path)
    inventory = tmp_path / "inventory.gz"
    module.write_inventory(tmp_path, inventory)
    path = tmp_path / "data/runs/response.txt"
    if change == "add":
        (tmp_path / "data/runs/new.txt").write_text("new")
    elif change == "edit":
        path.write_text("changed")
    elif change == "remove":
        path.unlink()
    else:
        path = tmp_path / "data/runs/python"
        path.unlink()
        path.symlink_to("/another/python")
    with pytest.raises(ValueError):
        module.verify(tmp_path, inventory)


def test_inventory_is_deterministic(tmp_path):
    tree(tmp_path)
    first, second = tmp_path / "one.gz", tmp_path / "two.gz"
    module.write_inventory(tmp_path, first)
    module.write_inventory(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()

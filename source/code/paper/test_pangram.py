"""Checks for text preservation and input boundaries."""

import json
import io
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest
import pangram

from pangram import clean_api, expand, groups, sha, windows


def test_nested_aux_groups():
    assert groups(r"{key}{{1}{2026{\em a}}{{Name}}}") == ["key", r"{1}{2026{\em a}}{{Name}}"]


def test_citations_and_references():
    text = r"A \citep{a,b}. \citet{a} says so (\Cref{fig:x,app:x})."
    result = expand(text, {"fig:x": "2", "app:x": "B"}, {"a": ("Author", "2025"), "b": ("Author", "2026")})
    assert result == "A (Author, 2025, 2026). Author (2025) says so (Figure~2 and Appendix~B)."


def test_exhibit_exclusion_preserves_surrounding_prose():
    text = r"Before. \begin{figure}\caption{Caption}\end{figure} After."
    assert expand(text, {}, {}) == "Before.  After."


def test_api_exclusions_preserve_inline_math():
    nodes = [{"t": "Header", "c": []}, {"t": "Note", "c": []},
             {"t": "Math", "c": [{"t": "DisplayMath"}, "x"]},
             {"t": "Math", "c": [{"t": "InlineMath"}, "y"]}]
    assert clean_api(nodes) == [nodes[-1]]


def test_windows_preserve_whitespace_and_cover_tail():
    text = "one  two\n\nthree four five six seven eight nine"
    spans = list(windows(text, size=4, step=2))
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(text[left:right] == value for left, right, value in spans)
    assert all(second[0] <= first[1] for first, second in zip(spans, spans[1:]))


def test_prepared_inputs_match_body():
    out = Path(__file__).resolve().parents[2] / "notes/pangram/2026-09-30-prepared"
    if not out.exists():
        return
    items = json.loads((out / "inputs.json").read_text())
    body = items[0]["text"]
    sections = [item for item in items if item["kind"] == "section"]
    assert len(sections) == 6
    assert "\n\n".join(item["text"] for item in sections) == body
    for item in items:
        assert sha(item["text"]) == item["sha256"]
        if "start_index" in item:
            assert body[item["start_index"]:item["end_index"]] == item["text"]


def test_payment_rejection_stops_without_polling(tmp_path, monkeypatch):
    text = "Example text."
    (tmp_path / "inputs.json").write_text(json.dumps([
        {"id": "whole_body", "text": text, "sha256": sha(text)},
        {"id": "section", "text": text, "sha256": sha(text)},
    ]))
    monkeypatch.setattr(pangram.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"test-credential"))
    calls = []
    def reject(req, **kwargs):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, None)
    monkeypatch.setattr(pangram.urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError, match="402"):
        pangram.run(tmp_path)
    assert calls == [pangram.API + "/task"]
    assert not list((tmp_path / "responses").glob("*.pending.json"))
    assert json.loads((tmp_path / "responses/whole_body.http_error.json").read_text())["status"] == 402


def prepare_mock_run(out, monkeypatch):
    """Keep detector tests independent of Keychain and the external API."""
    text = "Example text."
    (out / "inputs.json").write_text(json.dumps([
        {"id": "abstract", "text": text, "sha256": sha(text)},
    ]))
    monkeypatch.setattr(pangram.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b"test-credential"))
    return text


def test_request_selects_pangram_4_and_records_model(tmp_path, monkeypatch):
    text = prepare_mock_run(tmp_path, monkeypatch)
    calls = []
    response = {"stage": "STAGE_SUCCESS", "version": "4.0", "text": text}

    def respond(req, **kwargs):
        calls.append(req)
        value = {"task_id": "mock-task"} if req.get_method() == "POST" else response
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(pangram.urllib.request, "urlopen", respond)
    pangram.run(tmp_path)
    assert len(calls) == 2
    assert json.loads(calls[0].data) == {
        "model": "pangram-4", "text": text, "public_dashboard_link": False,
    }
    assert calls[1].full_url == pangram.API + "/task/mock-task"
    assert json.loads((tmp_path / "run_config.json").read_text())["model"] == "pangram-4"
    assert json.loads((tmp_path / "responses/abstract.json").read_text()) == response
    assert json.loads((tmp_path / "responses/abstract.text_validation.json").read_text()) == {
        "exact_match": True, "whitespace_equivalent": True,
        "submitted_sha256": sha(text), "returned_sha256": sha(text),
    }


def test_old_live_response_cannot_be_accepted_as_pangram_4(tmp_path, monkeypatch):
    text = prepare_mock_run(tmp_path, monkeypatch)

    def respond(req, **kwargs):
        value = ({"task_id": "mock-task"} if req.get_method() == "POST" else
                 {"stage": "STAGE_SUCCESS", "version": "3.3.2", "text": text})
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(pangram.urllib.request, "urlopen", respond)
    with pytest.raises(ValueError, match="(?i)version|model"):
        pangram.run(tmp_path)
    assert not (tmp_path / "responses/abstract.json").exists()


def test_old_cached_response_cannot_be_accepted_as_pangram_4(tmp_path, monkeypatch):
    text = prepare_mock_run(tmp_path, monkeypatch)
    (tmp_path / "run_config.json").write_text(json.dumps({"model": "pangram-4"}))
    (tmp_path / "responses").mkdir()
    cached = {"stage": "STAGE_SUCCESS", "version": "3.3.2", "text": text}
    (tmp_path / "responses/abstract.json").write_text(json.dumps(cached))

    def unexpected_request(*args, **kwargs):
        pytest.fail("A cached version mismatch must not trigger an API request")

    monkeypatch.setattr(pangram.urllib.request, "urlopen", unexpected_request)
    with pytest.raises(ValueError, match="(?i)version|model"):
        pangram.run(tmp_path)
    assert json.loads((tmp_path / "responses/abstract.json").read_text()) == cached


def test_conflicting_run_model_stops_before_network(tmp_path, monkeypatch):
    prepare_mock_run(tmp_path, monkeypatch)
    config = {"model": "pangram-3"}
    (tmp_path / "run_config.json").write_text(json.dumps(config))

    def unexpected_request(*args, **kwargs):
        pytest.fail("Conflicting model configuration must not trigger an API request")

    monkeypatch.setattr(pangram.urllib.request, "urlopen", unexpected_request)
    with pytest.raises((ValueError, RuntimeError), match="(?i)model"):
        pangram.run(tmp_path)
    assert json.loads((tmp_path / "run_config.json").read_text()) == config


def test_text_validation_accepts_collapsed_paragraph_separators():
    submitted = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    returned = submitted.replace("\n\n", "\n")
    assert pangram.validate_text(submitted, returned) == {
        "exact_match": False, "whitespace_equivalent": True,
        "submitted_sha256": sha(submitted), "returned_sha256": sha(returned),
    }


@pytest.mark.parametrize("returned", [
    "Task success rises to 79 percent.",
    "Task success rises to 97 percent!",
    "Tasksuccess rises to 97 percent.",
])
def test_text_validation_rejects_word_number_and_punctuation_changes(returned):
    with pytest.raises(ValueError):
        pangram.validate_text("Task success rises to 97 percent.", returned)


@pytest.mark.parametrize("cached", [False, True])
def test_runner_accepts_and_records_whitespace_changes(tmp_path, monkeypatch, cached):
    text = prepare_mock_run(tmp_path, monkeypatch)
    returned = text.replace(" ", "\n")
    response = {"stage": "STAGE_SUCCESS", "version": "4.0", "text": returned}
    if cached:
        (tmp_path / "responses").mkdir()
        (tmp_path / "responses/abstract.json").write_text(json.dumps(response))
        (tmp_path / "run_config.json").write_text(json.dumps({"model": "pangram-4"}))

    def respond(req, **kwargs):
        if cached:
            pytest.fail("Cached response validation must not trigger an API request")
        value = {"task_id": "mock-task"} if req.get_method() == "POST" else response
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(pangram.urllib.request, "urlopen", respond)
    pangram.run(tmp_path)
    assert json.loads((tmp_path / "responses/abstract.text_validation.json").read_text()) == {
        "exact_match": False, "whitespace_equivalent": True,
        "submitted_sha256": sha(text), "returned_sha256": sha(returned),
    }
    assert json.loads((tmp_path / "responses/abstract.json").read_text()) == response


@pytest.mark.parametrize("cached", [False, True])
def test_runner_rejects_changed_wording(tmp_path, monkeypatch, cached):
    prepare_mock_run(tmp_path, monkeypatch)
    response = {"stage": "STAGE_SUCCESS", "version": "4.0", "text": "Changed text."}
    if cached:
        (tmp_path / "responses").mkdir()
        (tmp_path / "responses/abstract.json").write_text(json.dumps(response))
        (tmp_path / "run_config.json").write_text(json.dumps({"model": "pangram-4"}))

    def respond(req, **kwargs):
        if cached:
            pytest.fail("Invalid cached text must not trigger an API request")
        value = {"task_id": "mock-task"} if req.get_method() == "POST" else response
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(pangram.urllib.request, "urlopen", respond)
    with pytest.raises(ValueError):
        pangram.run(tmp_path)
    if not cached:
        assert not (tmp_path / "responses/abstract.json").exists()

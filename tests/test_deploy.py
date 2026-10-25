"""Deployment templates: placeholders must be filled in before anything is applied."""

from __future__ import annotations

import importlib.util

from .conftest import ROOT


def _check():
    spec = importlib.util.spec_from_file_location("check_manifests",
                                                  ROOT / "deploy" / "check_manifests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_job_manifests_are_templates_and_are_rejected(capsys):
    mod = _check()
    jobs = [str(p) for p in sorted((ROOT / "deploy" / "k8s").glob("job-*.yaml"))]
    assert jobs and mod.main(jobs) == 1
    assert "REPLACE_GROUP" in capsys.readouterr().err


def test_filled_manifest_passes(tmp_path):
    mod = _check()
    src = (ROOT / "deploy" / "k8s" / "job-validate.yaml").read_text(encoding="utf-8")
    filled = src.replace("REPLACE_GROUP", "lab").replace("REPLACE_PROJECT", "heterofl") \
                .replace("REPLACE_TAG", "abc1234")
    (tmp_path / "job.yaml").write_text(filled, encoding="utf-8")
    assert mod.main([str(tmp_path / "job.yaml")]) == 0


def test_run_unit_refuses_placeholder_image(monkeypatch, capsys):
    import sys

    import pytest

    monkeypatch.setenv("IMAGE_REF", "REPLACE_GROUP/REPLACE_PROJECT:REPLACE_TAG")
    spec = importlib.util.spec_from_file_location("run_unit", ROOT / "deploy" / "run_unit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["run_unit.py"])
    with pytest.raises(SystemExit):
        mod.main(["--config", str(ROOT / "configs" / "paper.yaml"), "--index", "0", "--tag", "t"])
    assert "placeholder" in capsys.readouterr().err

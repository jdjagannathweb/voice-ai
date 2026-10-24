from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "voxcpm" / "model" / "compile_fallback.py"

spec = importlib.util.spec_from_file_location("voxcpm.model.compile_fallback", MODULE_PATH)
compile_fallback = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compile_fallback)


def test_compiler_found_via_path(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("CC", raising=False)
    assert compile_fallback.compiler_available() is True


def test_compiler_found_via_cc_env(monkeypatch):
    monkeypatch.setattr(
        compile_fallback.shutil,
        "which",
        lambda name: "/opt/my-cc" if name == "my-cc" else None,
    )
    monkeypatch.setenv("CC", "my-cc --flag")
    assert compile_fallback.compiler_available() is True


def test_compiler_found_via_triton_knobs_impl(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: None)
    monkeypatch.delenv("CC", raising=False)
    triton_stub = types.ModuleType("triton")
    knobs = types.SimpleNamespace(build=types.SimpleNamespace(impl="custom-cc -v"))
    triton_stub.knobs = knobs
    monkeypatch.setitem(sys.modules, "triton", triton_stub)
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: "/x/custom-cc" if name == "custom-cc" else None)
    assert compile_fallback.compiler_available() is True


def test_no_compiler_anywhere(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    assert compile_fallback.compiler_available() is False


def test_missing_triton_import_is_tolerated(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: "/usr/bin/gcc" if name == "gcc" else None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", None)  # import triton -> ImportError
    assert compile_fallback.compiler_available() is True

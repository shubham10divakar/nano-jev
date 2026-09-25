from pathlib import Path

import pytest

from nanojev import registry


@pytest.mark.parametrize("name,expected", [
    ("v0.1", True), ("v1.0", True), ("v10.2.3", True),
    ("0.1", False), ("v", False), ("sdmlai/nano-jev", False), ("runs/nano-jev-v0.1", False),
])
def test_is_version(name, expected):
    assert registry.is_version(name) is expected


def test_split_model():
    assert registry.split_model("v0.1") == (registry.DEFAULT_REPO, "v0.1")
    assert registry.split_model("someone/model@v2.0") == ("someone/model", "v2.0")
    assert registry.split_model("someone/model") == ("someone/model", None)


def test_selection_order(isolated_home, monkeypatch):
    assert registry.get_selected() == (registry.DEFAULT_VERSION, "default")

    registry.set_selected("v0.2")
    model, origin = registry.get_selected()
    assert model == "v0.2" and origin.endswith("config.json")

    monkeypatch.setenv("NANOJEV_MODEL", "v9.9")  # environment beats the saved choice
    assert registry.get_selected() == ("v9.9", "NANOJEV_MODEL")

    monkeypatch.delenv("NANOJEV_MODEL")
    registry.set_selected(None)
    assert registry.get_selected() == (registry.DEFAULT_VERSION, "default")


def test_local_folder_saved_as_absolute_path(isolated_home, tiny_model_dir, monkeypatch):
    monkeypatch.chdir(tiny_model_dir.parent)
    registry.set_selected(tiny_model_dir.name)
    assert Path(registry.get_selected()[0]) == tiny_model_dir.resolve()


def test_resolve_local_folder_needs_no_network(tiny_model_dir):
    assert registry.resolve(str(tiny_model_dir)) == tiny_model_dir


def test_list_models_finds_local_folders(tiny_model_dir):
    models, _ = registry.list_models(local_dirs=[tiny_model_dir.parent], online=False)
    local = [m for m in models if m.source == "local"]
    assert any(Path(m.name) == tiny_model_dir for m in local)

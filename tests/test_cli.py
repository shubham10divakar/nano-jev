import json

import pytest

import nanojev
from nanojev.__main__ import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert nanojev.__version__ in capsys.readouterr().out


def test_use_local_folder_then_current(isolated_home, tiny_model_dir, capsys):
    main(["use", str(tiny_model_dir)])
    main(["current"])
    out = capsys.readouterr().out
    assert str(tiny_model_dir.resolve()) in out


def test_use_reset(isolated_home, tiny_model_dir, capsys):
    main(["use", str(tiny_model_dir)])
    main(["use", "--reset"])
    assert f"using {nanojev.DEFAULT_VERSION}" in capsys.readouterr().out


def test_relevance_json(tiny_model_dir, capsys):
    main(["relevance", "--model", str(tiny_model_dir), "--json",
          "-q", "When was UCL founded?", "-p", "UCL was founded in 1826.", "-p", "London."])
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 2
    assert set(result[0]) == {"irrelevant", "partially relevant", "directly answers"}


def test_decide_json(tiny_model_dir, capsys):
    main(["decide", "--model", str(tiny_model_dir), "--json", "--question", "Which topic?",
          "-o", "history", "-o", "cooking", "--state", "UCL was founded in 1826."])
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"history", "cooking"}
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)


def test_list_offline_shows_local_folder(tiny_model_dir, isolated_home, capsys):
    main(["list", "--offline", "--local", str(tiny_model_dir.parent)])
    out = capsys.readouterr().out
    assert tiny_model_dir.name in out
    assert "selected:" in out

"""Shared fixtures: a tiny random model folder (no download) and an isolated config home."""

import json

import pytest

WORDS = ["question", "option", "how", "relevant", "is", "this", "passage", "to", "the", "query",
         "does", "context", "contain", "enough", "information", "answer", "claim", "supported",
         "by", "yes", "no", "irrelevant", "partially", "directly", "answers", "ucl", "was",
         "founded", "in", "1826", "1900", "when", "university", "college", "london", ":", "?", "."]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A 1-layer BERT cross-encoder saved like a real Nano-Jev release folder."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    folder = tmp_path_factory.mktemp("tiny-nano-jev")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (folder / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizer(vocab_file=str(folder / "vocab.txt")).save_pretrained(folder)

    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, num_labels=1)
    BertForSequenceClassification(config).save_pretrained(folder)

    (folder / "nanojev_config.json").write_text(json.dumps(
        {"model_name": "nano-jev", "version": "0.0-test", "max_length": 128}))
    (folder / "calibration.json").write_text(json.dumps(
        {"relevance": 1.5, "sufficient": 1.2, "grounded": 1.1}))
    return folder


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Keep `nano-jev use` choices out of the real ~/.nanojev."""
    monkeypatch.setenv("NANOJEV_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NANOJEV_MODEL", raising=False)
    return tmp_path / "home"


@pytest.fixture(scope="session")
def decider(tiny_model_dir):
    from nanojev import Decider
    return Decider.from_pretrained(str(tiny_model_dir), device="cpu")

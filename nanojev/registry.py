"""Model registry: list, download and select Nano-Jev weights.

Published versions are git tags ("v0.1", "v1.0", ...) on one Hugging Face repo. A model
can be named as:

- a version tag            "v0.1"                   -> DEFAULT_REPO at that tag
- a Hub repo, optional tag "user/nano-jev@v0.1"
- a local folder           "runs/nano-jev-v0.1"

When no model is given, the selection is: $NANOJEV_MODEL, then the model saved with
`python -m nanojev use`, then DEFAULT_VERSION.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "sdmlai/nano-jev"
DEFAULT_VERSION = "v0.1"  # pinned per package release, so results stay reproducible

_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$")


def is_version(name: str) -> bool:
    return bool(_VERSION_RE.match(name))


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in tag[1:].split("."))


@dataclass
class ModelInfo:
    name: str  # version tag, or local folder
    source: str  # "hub" | "local"
    status: str = ""
    base: str = ""
    params: str = ""
    released: str = ""
    downloaded: bool = False


def _read_config(folder: Path) -> dict:
    p = Path(folder) / "nanojev_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _info(cfg: dict, **fields) -> ModelInfo:
    return ModelInfo(status=cfg.get("status", ""), base=cfg.get("base", ""),
                     params=cfg.get("params", ""), released=cfg.get("released", ""), **fields)


# ------------------------------------------------------------------ discovery

def cached_versions(repo: str = DEFAULT_REPO) -> dict[str, Path]:
    """Version tags whose full weights are already in the local Hugging Face cache."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception:  # no cache directory yet
        return {}
    found = {}
    for r in cache.repos:
        if r.repo_id != repo or r.repo_type != "model":
            continue
        for rev in r.revisions:
            # Listing fetches only nanojev_config.json, so require the weights file too.
            if not (Path(rev.snapshot_path) / "model.safetensors").exists():
                continue
            for ref in rev.refs:
                if is_version(ref):
                    found[ref] = Path(rev.snapshot_path)
    return found


def hub_versions(repo: str = DEFAULT_REPO) -> list[str]:
    """Version tags published on the Hub, oldest first. Needs network."""
    from huggingface_hub import HfApi

    tags = [t.name for t in HfApi().list_repo_refs(repo).tags if is_version(t.name)]
    return sorted(tags, key=_version_key)


def list_models(repo: str = DEFAULT_REPO, local_dirs=(), online: bool = True):
    """Return (models, online_ok).

    Hub versions come first (oldest to newest), then any local folders found under
    `local_dirs`. When offline, only versions already in the cache are listed.
    """
    cached = cached_versions(repo)
    infos: dict[str, ModelInfo] = {}
    online_ok = False
    if online:
        try:
            from huggingface_hub import hf_hub_download

            for tag in hub_versions(repo):
                cfg = json.loads(Path(hf_hub_download(repo, "nanojev_config.json",
                                                      revision=tag)).read_text(encoding="utf-8"))
                infos[tag] = _info(cfg, name=tag, source="hub", downloaded=tag in cached)
            online_ok = True
        except Exception:
            pass
    for tag, path in cached.items():
        infos.setdefault(tag, _info(_read_config(path), name=tag, source="hub", downloaded=True))
    models = [infos[t] for t in sorted(infos, key=_version_key)]

    for d in local_dirs:
        root = Path(d)
        for folder in [root, *sorted(p for p in root.glob("*") if p.is_dir())]:
            if (folder / "model.safetensors").exists():
                models.append(_info(_read_config(folder), name=str(folder), source="local",
                                    downloaded=True))
    return models, online_ok


# ------------------------------------------------------------------ selection

def _config_path() -> Path:
    return Path(os.environ.get("NANOJEV_HOME", Path.home() / ".nanojev")) / "config.json"


def get_selected() -> tuple[str, str]:
    """Return (model, where the choice came from)."""
    if os.environ.get("NANOJEV_MODEL"):
        return os.environ["NANOJEV_MODEL"], "NANOJEV_MODEL"
    p = _config_path()
    if p.exists():
        model = json.loads(p.read_text(encoding="utf-8")).get("model")
        if model:
            return model, str(p)
    return DEFAULT_VERSION, "default"


def set_selected(model: str | None) -> Path:
    """Save the model to use by default; None clears the choice."""
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if model is None:
        cfg.pop("model", None)
    else:
        cfg["model"] = str(Path(model).resolve()) if Path(model).is_dir() else model
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


def split_model(model: str, repo: str = DEFAULT_REPO) -> tuple[str, str | None]:
    """"v0.1" -> (repo, "v0.1"); "user/x@v0.1" -> ("user/x", "v0.1"); "user/x" -> ("user/x", None)."""
    if is_version(model):
        return repo, model
    repo_id, _, revision = model.partition("@")
    return repo_id, revision or None


def resolve(model: str | None = None, download: bool = True) -> Path:
    """Turn a model name (see module docstring) into a local folder, downloading if needed."""
    model = model or get_selected()[0]
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    repo_id, revision = split_model(model)
    path = Path(snapshot_download(repo_id, revision=revision, local_files_only=not download))
    # `list` caches only nanojev_config.json, which leaves a partial snapshot behind.
    if not download and not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"weights for {model} are not downloaded")
    return path

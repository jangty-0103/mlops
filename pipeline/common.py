"""파이프라인 공통 도구: 경로, 설정, 해시, Git 상태, 실행 환경."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
RAW_ZIP = ROOT / "data" / "raw" / "sms_spam_collection.zip"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"
MODEL_PATH = ROOT / "artifacts" / "model" / "model.json"
REPORT_DIR = ROOT / "artifacts" / "reports"
RUN_LOG = ROOT / "runs" / "run_log.jsonl"


class StepFailed(RuntimeError):
    """검사에 실패해 다음 단계를 중단해야 할 때. 그때까지의 결과를 함께 담는다."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return read_json(CONFIG_PATH)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_state() -> dict:
    """현재 커밋과, 커밋되지 않은 변경이 있는지 여부."""

    def git(*args):
        try:
            done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}


def environment() -> dict:
    packages = {}
    for name in ("scikit-learn", "numpy", "scipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }

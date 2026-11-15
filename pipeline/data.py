"""데이터 준비: 내려받기 -> 잘못된 값 처리 -> 중복 제거 -> 학습용/평가용 분리."""
from __future__ import annotations

import html
import urllib.request
import zipfile
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

from . import common

VALID_LABELS = ("ham", "spam")
SOURCE = {
    "name": "SMS Spam Collection (UCI Machine Learning Repository)",
    "page": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    "doi": "10.24432/C5CC84",
    "license": "CC BY 4.0",
}


def _download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        f.write(response.read())
    tmp.replace(dest)


def _parse(zip_path, member: str) -> tuple[list[dict], dict]:
    """한 줄이 '정답<TAB>문자 내용'인 원본을 읽는다. 형식이 틀린 줄은 버리고 센다."""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member).decode("utf-8", errors="replace")
    rows = []
    stats = Counter()
    # splitlines()는 문자 내용 안의 특수 줄바꿈 문자까지 나누므로 "\n"으로만 나눈다.
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        stats["raw_lines"] += 1
        label, tab, text = line.partition("\t")
        if not tab or label not in VALID_LABELS:
            stats["rejected_bad_format"] += 1
            continue
        # 원본에 남아 있는 HTML 표기(&lt; &amp; 등)를 실제 문자로 되돌린다.
        text = html.unescape(text).strip()
        if not text:
            stats["rejected_empty_text"] += 1
            continue
        if "�" in text:
            stats["rows_with_undecodable_chars"] += 1
        rows.append({"label": label, "text": text})
    return rows, dict(stats)


def _deduplicate(rows: list[dict]) -> tuple[list[dict], dict]:
    """내용이 같은 문자는 하나만 남긴다. 정답이 서로 다르면 모두 버린다."""
    labels_by_text = defaultdict(set)
    for row in rows:
        labels_by_text[row["text"]].add(row["label"])
    kept, seen = [], set()
    stats = Counter()
    for row in rows:
        text = row["text"]
        if len(labels_by_text[text]) > 1:
            stats["dropped_conflicting_label_rows"] += 1
        elif text in seen:
            stats["dropped_duplicate_rows"] += 1
        else:
            seen.add(text)
            kept.append(row)
    return kept, dict(stats)


def run(config: dict, options: dict) -> dict:
    cfg = config["data"]
    downloaded = False
    if not common.RAW_ZIP.exists():
        _download(cfg["url"], common.RAW_ZIP)
        downloaded = True
    raw_sha256 = common.sha256_file(common.RAW_ZIP)
    if cfg.get("sha256") and raw_sha256 != cfg["sha256"]:
        raise ValueError(
            f"원본 데이터의 sha256이 설정과 다릅니다: {raw_sha256} (설정: {cfg['sha256']})"
        )

    rows, parse_stats = _parse(common.RAW_ZIP, cfg["member"])
    rows, dedup_stats = _deduplicate(rows)

    # 중복을 먼저 없앤 뒤 나누므로 같은 문자가 학습용과 평가용에 함께 들어가지 않는다.
    train, test = train_test_split(
        rows,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
        shuffle=True,
        stratify=[row["label"] for row in rows],
    )
    overlap = len({r["text"] for r in train} & {r["text"] for r in test})
    if overlap:
        raise ValueError(f"학습용과 평가용에 같은 문자가 {overlap}건 있습니다")

    common.write_jsonl(common.TRAIN_PATH, train)
    common.write_jsonl(common.TEST_PATH, test)

    summary = {
        "source": SOURCE,
        "downloaded_this_run": downloaded,
        "raw_sha256": raw_sha256,
        **parse_stats,
        **dedup_stats,
        "rows_after_cleaning": len(rows),
        "class_counts": dict(Counter(r["label"] for r in rows)),
        "train_rows": len(train),
        "train_class_counts": dict(Counter(r["label"] for r in train)),
        "test_rows": len(test),
        "test_class_counts": dict(Counter(r["label"] for r in test)),
        "train_test_overlap": overlap,
        "train_sha256": common.sha256_file(common.TRAIN_PATH),
        "test_sha256": common.sha256_file(common.TEST_PATH),
    }
    report_path = common.REPORT_DIR / "data_report.json"
    common.write_json(report_path, summary)
    return {
        "inputs": {"url": cfg["url"], "raw": common.rel(common.RAW_ZIP), "raw_sha256": raw_sha256},
        "outputs": {
            "train": common.rel(common.TRAIN_PATH),
            "test": common.rel(common.TEST_PATH),
            "report": common.rel(report_path),
        },
        "summary": {
            "rows_after_cleaning": len(rows),
            "train_rows": len(train),
            "test_rows": len(test),
        },
    }

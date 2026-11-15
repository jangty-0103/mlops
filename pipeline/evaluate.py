"""평가: 서비스가 실제로 쓰는 모델 파일로 평가용 자료를 예측하고 기준 방법과 비교한다.

기준 방법: 학습용 자료에서 가장 많은 종류(정상)를 항상 답한다.
평가용 자료는 학습과 설정 선택에 쓰지 않았고, 여기서 처음 읽는다.
"""
from __future__ import annotations

from collections import Counter

from spamsvc.model import load_model_file

from . import common

POSITIVE = "spam"


def _metrics(truth: list[str], predicted: list[str]) -> dict:
    tp = sum(t == POSITIVE and p == POSITIVE for t, p in zip(truth, predicted))
    fp = sum(t != POSITIVE and p == POSITIVE for t, p in zip(truth, predicted))
    fn = sum(t == POSITIVE and p != POSITIVE for t, p in zip(truth, predicted))
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round((tp + tn) / len(truth), 4),
        "spam_precision": round(precision, 4),
        "spam_recall": round(recall, 4),
        "spam_f1": round(f1, 4),
        "confusion": {"spam_as_spam": tp, "ham_as_spam": fp, "spam_as_ham": fn, "ham_as_ham": tn},
    }


def run(config: dict, options: dict) -> dict:
    model = load_model_file(common.MODEL_PATH)
    train_sha256 = common.sha256_file(common.TRAIN_PATH)
    if model.training.get("train_sha256") != train_sha256:
        raise common.StepFailed("모델이 현재 학습용 자료로 만들어진 것이 아닙니다. train 단계를 다시 실행하세요.")

    train = common.read_jsonl(common.TRAIN_PATH)
    test = common.read_jsonl(common.TEST_PATH)
    truth = [row["label"] for row in test]

    probabilities = [model.spam_probability(row["text"]) for row in test]
    predicted = [POSITIVE if p >= model.threshold else "ham" for p in probabilities]

    majority = Counter(row["label"] for row in train).most_common(1)[0][0]
    baseline = [majority] * len(test)

    errors = []
    for index, (row, p, guess) in enumerate(zip(test, probabilities, predicted)):
        if guess != row["label"]:
            errors.append({
                "test_index": index,
                "true": row["label"],
                "predicted": guess,
                "spam_probability": round(p, 4),
                "text": row["text"],
                "top_tokens": [[t, round(c, 3)] for t, c in model.contributions(row["text"])[:8]],
            })
    # 확신이 컸는데 틀린 것부터 본다.
    errors.sort(key=lambda e: abs(e["spam_probability"] - model.threshold), reverse=True)
    errors_path = common.REPORT_DIR / "eval_errors.jsonl"
    common.write_jsonl(errors_path, errors)

    report = {
        "model_version": model.version,
        "threshold": model.threshold,
        "test_rows": len(test),
        "test_class_counts": dict(Counter(truth)),
        "test_sha256": common.sha256_file(common.TEST_PATH),
        "model": _metrics(truth, predicted),
        "baseline_majority": {"always_answers": majority, **_metrics(truth, baseline)},
        "error_count": len(errors),
        "errors_file": common.rel(errors_path),
    }
    report_path = common.REPORT_DIR / "eval_report.json"
    common.write_json(report_path, report)
    return {
        "model_version": model.version,
        "inputs": {
            "model": common.rel(common.MODEL_PATH),
            "test": common.rel(common.TEST_PATH),
            "test_sha256": report["test_sha256"],
        },
        "outputs": {"report": common.rel(report_path), "errors": common.rel(errors_path)},
        "summary": {
            "model_spam_f1": report["model"]["spam_f1"],
            "baseline_spam_f1": report["baseline_majority"]["spam_f1"],
            "model_accuracy": report["model"]["accuracy"],
            "baseline_accuracy": report["baseline_majority"]["accuracy"],
            "error_count": len(errors),
        },
    }

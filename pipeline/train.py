"""학습: TF-IDF + 로지스틱 회귀를 학습하고, 서비스가 읽는 JSON 모델 파일로 내보낸다.

평가용 자료는 이 단계에서 읽지 않는다. 설정값(C)은 학습용 자료 안의 교차검증으로만 고른다.
"""
from __future__ import annotations

import json

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from spamsvc.model import LABELS, MODEL_FORMAT, load_model_bytes, tokenize

from . import common

PARITY_TOLERANCE = 1e-9


def run(config: dict, options: dict) -> dict:
    cfg = config["train"]
    rows = common.read_jsonl(common.TRAIN_PATH)
    texts = [row["text"] for row in rows]
    y = [LABELS.index(row["label"]) for row in rows]  # ham=0, spam=1
    train_sha256 = common.sha256_file(common.TRAIN_PATH)

    pipe = Pipeline([
        # 서비스와 같은 tokenize를 써서 학습과 예측의 전처리를 일치시킨다.
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                  min_df=cfg["min_df"])),
        ("clf", LogisticRegression(class_weight=cfg["class_weight"], max_iter=cfg["max_iter"])),
    ])
    folds = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    search = GridSearchCV(pipe, {"clf__C": cfg["c_grid"]}, scoring="f1", cv=folds)
    search.fit(texts, y)

    best = search.best_estimator_
    vectorizer, classifier = best["tfidf"], best["clf"]
    coef = classifier.coef_[0]
    tokens = {
        token: [float(vectorizer.idf_[index]), float(coef[index])]
        for token, index in vectorizer.vocabulary_.items()
    }
    model = {
        "format": MODEL_FORMAT,
        "labels": list(LABELS),
        "threshold": config["serve"]["threshold"],
        "intercept": float(classifier.intercept_[0]),
        "tokens": tokens,
        "training": {
            "train_sha256": train_sha256,
            "train_rows": len(rows),
            "C": search.best_params_["clf__C"],
            "min_df": cfg["min_df"],
            "class_weight": cfg["class_weight"],
            "sklearn_version": sklearn.__version__,
        },
    }
    # 같은 자료와 설정이면 같은 파일(같은 버전)이 나오도록 시각은 넣지 않고 키를 정렬한다.
    raw = (json.dumps(model, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    exported = load_model_bytes(raw)

    # 내보낸 모델의 순수 파이썬 계산이 scikit-learn 결과와 같은지 확인한다.
    sk_probabilities = best.predict_proba(texts)[:, 1]
    max_diff = max(
        abs(exported.spam_probability(text) - float(p)) for text, p in zip(texts, sk_probabilities)
    )
    if max_diff > PARITY_TOLERANCE:
        raise common.StepFailed(
            f"내보낸 모델과 scikit-learn의 확률이 다릅니다 (최대 차이 {max_diff:.3e})"
        )

    common.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.MODEL_PATH.write_bytes(raw)

    cv = search.cv_results_
    report = {
        "model_version": exported.version,
        "model_file": common.rel(common.MODEL_PATH),
        "model_bytes": len(raw),
        "vocabulary_size": len(tokens),
        "train_rows": len(rows),
        "train_sha256": train_sha256,
        "best_C": search.best_params_["clf__C"],
        "cv_f1_by_C": [
            {"C": c, "mean_f1": round(float(m), 4), "std_f1": round(float(s), 4)}
            for c, m, s in zip(cv["param_clf__C"].tolist(), cv["mean_test_score"], cv["std_test_score"])
        ],
        "parity_max_abs_diff": max_diff,
        "parity_tolerance": PARITY_TOLERANCE,
    }
    report_path = common.REPORT_DIR / "train_report.json"
    common.write_json(report_path, report)
    return {
        "model_version": exported.version,
        "inputs": {"train": common.rel(common.TRAIN_PATH), "train_sha256": train_sha256},
        "outputs": {"model": common.rel(common.MODEL_PATH), "report": common.rel(report_path)},
        "summary": {
            "best_C": report["best_C"],
            "vocabulary_size": len(tokens),
            "parity_max_abs_diff": max_diff,
        },
    }

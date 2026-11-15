"""순수 파이썬 스팸 분류 추론.

표준 라이브러리만 사용하므로 Lambda에 별도 패키지 없이 올릴 수 있다.
학습(pipeline/train.py)도 여기의 tokenize를 그대로 써서 학습과 서비스의 전처리가 같다.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

MODEL_FORMAT = "tfidf-logreg-v1"
LABELS = ("ham", "spam")

_TOKEN_RE = re.compile(r"\b\w\w+\b")


class ModelFormatError(ValueError):
    """모델 파일의 형식이 기대와 다를 때."""


def tokenize(text: str) -> list[str]:
    """소문자로 바꾼 뒤 두 글자 이상의 단어만 뽑는다."""
    return _TOKEN_RE.findall(text.lower())


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class SpamModel:
    version: str  # 모델 파일 sha256의 앞 12자리
    threshold: float
    intercept: float
    tokens: dict  # 단어 -> (idf, 가중치)
    training: dict

    def contributions(self, text: str) -> list[tuple[str, float]]:
        """단어별 점수 기여도. 절댓값이 큰 순서로 돌려준다."""
        counts = Counter(t for t in tokenize(text) if t in self.tokens)
        weights = {t: n * self.tokens[t][0] for t, n in counts.items()}
        norm = math.sqrt(math.fsum(w * w for w in weights.values()))
        if norm == 0.0:
            return []
        items = [(t, w / norm * self.tokens[t][1]) for t, w in weights.items()]
        return sorted(items, key=lambda item: abs(item[1]), reverse=True)

    def spam_probability(self, text: str) -> float:
        z = self.intercept + math.fsum(c for _, c in self.contributions(text))
        return _sigmoid(z)

    def predict(self, text: str) -> dict:
        probability = self.spam_probability(text)
        return {
            "label": LABELS[1] if probability >= self.threshold else LABELS[0],
            "spam_probability": round(probability, 6),
            "threshold": self.threshold,
        }


def model_version(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:12]


def load_model_bytes(raw: bytes) -> SpamModel:
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ModelFormatError(f"모델 파일을 JSON으로 읽을 수 없습니다: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
        raise ModelFormatError(f"지원하지 않는 모델 형식입니다 (필요: {MODEL_FORMAT})")
    try:
        tokens = {t: (float(v[0]), float(v[1])) for t, v in data["tokens"].items()}
        return SpamModel(
            version=model_version(raw),
            threshold=float(data["threshold"]),
            intercept=float(data["intercept"]),
            tokens=tokens,
            training=dict(data.get("training", {})),
        )
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ModelFormatError(f"모델 파일에 필요한 값이 없습니다: {exc!r}") from exc


def load_model_file(path) -> SpamModel:
    with open(path, "rb") as f:
        return load_model_bytes(f.read())

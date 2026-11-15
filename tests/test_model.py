import json
import math

import pytest

from spamsvc.model import ModelFormatError, load_model_bytes, tokenize

MODEL = {
    "format": "tfidf-logreg-v1",
    "labels": ["ham", "spam"],
    "threshold": 0.5,
    "intercept": -1.0,
    "tokens": {"free": [2.0, 3.0], "prize": [1.5, 2.0], "lunch": [1.0, -2.0]},
    "training": {},
}


def raw(model=MODEL) -> bytes:
    return json.dumps(model).encode("utf-8")


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def test_tokenize_lowercases_and_drops_single_characters():
    assert tokenize("FREE prize, U r a WINNER!") == ["free", "prize", "winner"]


def test_probability_matches_hand_calculation():
    model = load_model_bytes(raw())
    # free: 2.0, prize: 1.5 -> 길이 2.5로 나눈 뒤 가중치를 곱해 더한다.
    expected = sigmoid(-1.0 + (2.0 / 2.5) * 3.0 + (1.5 / 2.5) * 2.0)
    assert model.spam_probability("FREE prize") == pytest.approx(expected)
    assert model.predict("FREE prize")["label"] == "spam"


def test_repeated_word_counts_twice():
    model = load_model_bytes(raw())
    expected = sigmoid(-1.0 + (4.0 * 3.0 + 1.0 * -2.0) / math.sqrt(17.0))
    assert model.spam_probability("free free lunch") == pytest.approx(expected)


def test_unknown_words_fall_back_to_intercept():
    model = load_model_bytes(raw())
    assert model.spam_probability("zzz qqq") == pytest.approx(sigmoid(-1.0))
    assert model.predict("zzz qqq")["label"] == "ham"
    assert model.contributions("zzz qqq") == []


def test_version_changes_when_file_changes():
    changed = dict(MODEL, intercept=-1.5)
    assert load_model_bytes(raw()).version != load_model_bytes(raw(changed)).version
    assert load_model_bytes(raw()).version == load_model_bytes(raw()).version


@pytest.mark.parametrize("bad", [
    b"not json",
    json.dumps({"format": "other"}).encode(),
    json.dumps({"format": "tfidf-logreg-v1", "threshold": 0.5}).encode(),
    json.dumps([1, 2, 3]).encode(),
])
def test_bad_model_file_is_rejected(bad):
    with pytest.raises(ModelFormatError):
        load_model_bytes(bad)

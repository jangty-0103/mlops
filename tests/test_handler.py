import base64
import json

import pytest

from spamsvc import handler
from tests.test_model import raw


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    path.write_bytes(raw())
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.delenv("MAX_CHARS", raising=False)
    handler._cache.clear()
    return path


def call(event):
    response = handler.lambda_handler(event, None)
    return response["statusCode"], json.loads(response["body"])


def http(payload):
    return {"body": json.dumps(payload)}


def test_http_request_returns_prediction(model_path):
    status, body = call(http({"text": "FREE prize"}))
    assert status == 200
    assert body["label"] == "spam"
    assert 0 <= body["spam_probability"] <= 1
    assert len(body["model_version"]) == 12
    assert body["request_id"]


def test_direct_invoke_returns_prediction(model_path):
    status, body = call({"text": "lunch"})
    assert (status, body["label"]) == (200, "ham")


def test_base64_body_is_decoded(model_path):
    encoded = base64.b64encode(json.dumps({"text": "FREE prize"}).encode()).decode()
    status, body = call({"body": encoded, "isBase64Encoded": True})
    assert (status, body["label"]) == (200, "spam")


@pytest.mark.parametrize("event, error", [
    (http({}), "missing_text"),
    (http({"text": "   "}), "empty_text"),
    (http({"text": 123}), "invalid_text_type"),
    (http({"text": "a" * 1001}), "text_too_long"),
    ({"body": "not json"}, "invalid_json"),
    ({"body": "[1, 2]"}, "invalid_json"),
    ({"body": None}, "invalid_json"),
])
def test_bad_input_returns_400(model_path, event, error):
    status, body = call(event)
    assert (status, body["error"]) == (400, error)


def test_max_chars_follows_environment(model_path, monkeypatch):
    monkeypatch.setenv("MAX_CHARS", "5")
    status, body = call(http({"text": "lunch time"}))
    assert (status, body["error"]) == (400, "text_too_long")


def test_wrong_model_path_returns_500_then_recovers(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path.with_name("missing.json")))
    status, body = call(http({"text": "FREE prize"}))
    assert (status, body["error"]) == (500, "model_unavailable")
    assert "label" not in body  # 실패한 결과는 정상 결과와 구분된다

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    status, body = call(http({"text": "FREE prize"}))
    assert (status, body["label"]) == (200, "spam")


def test_missing_model_settings_returns_500(model_path, monkeypatch):
    monkeypatch.delenv("MODEL_PATH")
    status, body = call(http({"text": "FREE prize"}))
    assert (status, body["error"]) == (500, "model_unavailable")

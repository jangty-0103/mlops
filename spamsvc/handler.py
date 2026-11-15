"""Lambda 핸들러: 문자 내용을 받아 스팸 여부를 돌려준다.

입력 형식 두 가지를 받는다.
- HTTP 요청(함수 URL, API Gateway): event["body"]에 JSON 문자열 {"text": "..."}
- 직접 호출(lambda invoke): event 자체가 {"text": "..."}

모델 파일 위치는 환경변수 MODEL_PATH로 정한다.
Lambda에서는 배포 묶음 안의 상대 경로(model/model.json)를 쓴다.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid

from .model import SpamModel, load_model_file

DEFAULT_MAX_CHARS = 1000

_cache: dict = {}


class BadRequest(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelLoadError(RuntimeError):
    """모델을 읽지 못했을 때."""


def _get_model() -> SpamModel:
    path = os.environ.get("MODEL_PATH")
    if not path:
        raise ModelLoadError("MODEL_PATH 환경변수가 필요합니다")
    if path not in _cache:
        try:
            _cache[path] = load_model_file(path)
        except Exception as exc:
            raise ModelLoadError(f"{type(exc).__name__}: {exc}") from exc
    return _cache[path]


def _payload(event) -> dict:
    if isinstance(event, dict) and "body" in event:
        body = event.get("body") or ""
        try:
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode("utf-8")
            data = json.loads(body)
        except ValueError as exc:
            raise BadRequest("invalid_json", "요청 본문이 JSON이 아닙니다") from exc
    else:
        data = event
    if not isinstance(data, dict):
        raise BadRequest("invalid_json", "요청은 JSON 객체여야 합니다")
    return data


def _text(data: dict) -> str:
    if "text" not in data:
        raise BadRequest("missing_text", "필수 입력값 text가 없습니다")
    text = data["text"]
    if not isinstance(text, str):
        raise BadRequest("invalid_text_type", "text는 문자열이어야 합니다")
    text = text.strip()
    if not text:
        raise BadRequest("empty_text", "text가 비어 있습니다")
    max_chars = int(os.environ.get("MAX_CHARS", DEFAULT_MAX_CHARS))
    if len(text) > max_chars:
        raise BadRequest("text_too_long", f"text는 {max_chars}자 이하여야 합니다")
    return text


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def _log(**fields) -> None:
    # 문자 내용은 남기지 않고 길이만 남긴다.
    print(json.dumps(fields, ensure_ascii=False))


def lambda_handler(event, context):
    request_id = getattr(context, "aws_request_id", None) or uuid.uuid4().hex
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    try:
        text = _text(_payload(event))
        model = _get_model()
        result = model.predict(text)
    except BadRequest as exc:
        _log(event="rejected", request_id=request_id, status=400, error=exc.code, ms=elapsed_ms())
        return _response(400, {"error": exc.code, "message": str(exc), "request_id": request_id})
    except ModelLoadError as exc:
        _log(event="failed", request_id=request_id, status=500, error="model_unavailable",
             detail=str(exc), ms=elapsed_ms())
        return _response(500, {"error": "model_unavailable",
                               "message": "모델을 불러오지 못했습니다", "request_id": request_id})
    except Exception as exc:  # 예상하지 못한 오류도 정상 결과와 구분되게 500으로 돌려준다
        _log(event="failed", request_id=request_id, status=500, error="internal_error",
             detail=f"{type(exc).__name__}: {exc}", ms=elapsed_ms())
        return _response(500, {"error": "internal_error",
                               "message": "처리 중 오류가 발생했습니다", "request_id": request_id})

    _log(event="predicted", request_id=request_id, status=200, model_version=model.version,
         label=result["label"], spam_probability=result["spam_probability"],
         chars=len(text), ms=elapsed_ms())
    return _response(200, {**result, "model_version": model.version, "request_id": request_id})

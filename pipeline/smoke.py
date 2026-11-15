"""서비스 동작 검사: 정상 입력과 잘못된 입력을 보내 응답 형식과 상태 코드를 확인한다.

모델 품질 평가(evaluate)와 별개로, 서비스가 입력을 받아 결과를 돌려주는지를 본다.
대상은 로컬 핸들러(기본), 배포된 Lambda 함수(--smoke-lambda), HTTP 주소(--smoke-url) 중 하나다.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from spamsvc.model import LABELS, load_model_file

from . import common


def _cases(max_chars: int) -> list[dict]:
    return [
        {"name": "normal_spam", "expect_status": 200, "expect_label": "spam",
         "payload": {"text": "WINNER!! You have won a 1000 cash prize. Call 09061701461 to claim now."}},
        {"name": "normal_ham", "expect_status": 200, "expect_label": "ham",
         "payload": {"text": "Hey, are we still meeting for lunch tomorrow?"}},
        {"name": "direct_invoke", "expect_status": 200, "direct": True,
         "payload": {"text": "Ok, see you at home later."}},
        {"name": "missing_text", "expect_status": 400, "expect_error": "missing_text", "payload": {}},
        {"name": "empty_text", "expect_status": 400, "expect_error": "empty_text",
         "payload": {"text": "   "}},
        {"name": "wrong_type", "expect_status": 400, "expect_error": "invalid_text_type",
         "payload": {"text": 123}},
        {"name": "too_long", "expect_status": 400, "expect_error": "text_too_long",
         "payload": {"text": "a" * (max_chars + 1)}},
        {"name": "invalid_json", "expect_status": 400, "expect_error": "invalid_json",
         "raw_body": "not json"},
    ]


def _body(case: dict) -> str:
    return case["raw_body"] if "raw_body" in case else json.dumps(case["payload"])


def _invoke_local(case: dict) -> tuple[int, dict]:
    from spamsvc import handler

    event = case["payload"] if case.get("direct") else {"body": _body(case)}
    response = handler.lambda_handler(event, None)
    return response["statusCode"], json.loads(response["body"])


def _invoke_url(url: str, case: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=_body(case).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, {"unparsed_body": raw.decode("utf-8", errors="replace")[:300]}


def _invoke_lambda(lam, name: str, case: dict) -> tuple[int, dict]:
    from cloud import aws

    event = case["payload"] if case.get("direct") else {"body": _body(case)}
    status, body, _ = aws.invoke_function(lam, name, event)
    return status, body


def _check(case: dict, status: int, body: dict, expected_version: str | None) -> list[str]:
    problems = []
    if status != case["expect_status"]:
        problems.append(f"상태 코드 {status} (기대: {case['expect_status']})")
        return problems
    if status == 200:
        if body.get("label") not in LABELS:
            problems.append(f"label 값이 잘못됨: {body.get('label')!r}")
        probability = body.get("spam_probability")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            problems.append(f"spam_probability 값이 잘못됨: {probability!r}")
        if "expect_label" in case and body.get("label") != case["expect_label"]:
            problems.append(f"label {body.get('label')!r} (기대: {case['expect_label']!r})")
        if expected_version and body.get("model_version") != expected_version:
            problems.append(f"model_version {body.get('model_version')!r} (기대: {expected_version!r})")
    elif body.get("error") != case.get("expect_error"):
        problems.append(f"error {body.get('error')!r} (기대: {case.get('expect_error')!r})")
    return problems


def run(config: dict, options: dict) -> dict:
    url = options.get("smoke_url")
    max_chars = config["serve"]["max_chars"]
    expected_version = load_model_file(common.MODEL_PATH).version

    saved = {key: os.environ.get(key) for key in ("MODEL_PATH", "MAX_CHARS")}
    hidden = options.setdefault("hidden", {})  # 기록에서 가릴 값 (계정 번호, 사용자 이름)
    if options.get("smoke_lambda"):
        from cloud import aws

        who = aws.identity(config)
        hidden.update(who["hidden"])
        cloud_config = aws.resolve(config, who["user"])
        lam = aws.client("lambda", cloud_config)
        name = cloud_config["cloud"]["function_name"]
        target = f"lambda:{name}"
        invoke = lambda case: _invoke_lambda(lam, name, case)
    elif url:
        target = url
        invoke = lambda case: _invoke_url(url, case)
    else:
        target = "local-handler"
        invoke = _invoke_local
        os.environ["MODEL_PATH"] = str(common.MODEL_PATH)
        os.environ["MAX_CHARS"] = str(max_chars)
    results = []
    try:
        for case in _cases(max_chars):
            if url and case.get("direct"):
                continue
            status, body = invoke(case)
            problems = _check(case, status, body, expected_version)
            request_preview = _body(case)
            results.append({
                "name": case["name"],
                "request": request_preview[:120] + ("..." if len(request_preview) > 120 else ""),
                "status": status,
                "response": body,
                "passed": not problems,
                "problems": problems,
            })
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    failed = [r["name"] for r in results if not r["passed"]]
    report = {
        "target": target,
        "expected_model_version": expected_version,
        "checked_at": common.utc_now(),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "cases": results,
    }
    # 클라우드 검사는 실행마다 따로 남긴다 (실패와 복구 기록을 모두 보존).
    if target == "local-handler":
        report_path = common.REPORT_DIR / "smoke_report_local.json"
    else:
        report_path = common.REPORT_DIR / "cloud_smoke" / f"{options['run_id']}.json"
    common.write_json(report_path, common.mask(report, hidden))
    result = {
        "model_version": expected_version,
        "inputs": {"target": target, "model": common.rel(common.MODEL_PATH)},
        "outputs": {"report": common.rel(report_path)},
        "summary": {"passed": report["passed"], "failed": report["failed"], "failed_cases": failed},
    }
    if failed:
        raise common.StepFailed(f"서비스 검사 실패: {', '.join(failed)}", result)
    return result

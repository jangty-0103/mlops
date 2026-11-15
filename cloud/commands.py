"""AWS 명령 본체. 각 함수는 실행 기록에 남길 dict를 돌려준다."""
from __future__ import annotations

import base64
import hashlib
import time

from botocore.exceptions import ClientError

from pipeline import common
from spamsvc.model import model_version

from . import aws

NOT_FOUND = {"ResourceNotFoundException", "NoSuchEntity"}


def _function_config(lam, name: str) -> dict | None:
    try:
        return lam.get_function_configuration(FunctionName=name)
    except ClientError as exc:
        if aws.error_code(exc) in NOT_FOUND:
            return None
        raise


def _log_group_info(logs, name: str) -> dict | None:
    groups = logs.describe_log_groups(logGroupNamePrefix=name).get("logGroups", [])
    return next((g for g in groups if g["logGroupName"] == name), None)


def _probe(call) -> str:
    try:
        call()
        return "ok"
    except ClientError as exc:
        return f"denied ({aws.error_code(exc)})"


def check(config: dict, args, who: dict) -> dict:
    """인증 정보와 필요한 서비스의 접근 권한을 확인한다. 자원은 만들지 않는다.

    읽기 권한만 확인할 수 있다. 만들기 권한은 deploy를 실행해야 알 수 있다.
    """
    cfg = config["cloud"]
    summary = {
        "credentials": "ok",
        "region": cfg["region"],
        "lambda_list": _probe(lambda: aws.client("lambda", config).list_functions(MaxItems=1)),
        "logs_list": _probe(lambda: aws.client("logs", config).describe_log_groups(limit=1)),
        "role": _probe(lambda: aws.client("iam", config).get_role(RoleName=cfg["role_name"])),
        "function_name": cfg["function_name"],
        "role_name": cfg["role_name"],
    }
    if any(str(value).startswith("denied") for value in summary.values()):
        summary["hint"] = "MFA가 필요한 계정이면 python -m cloud.mfa 실행 후 AWS_PROFILE=mlops-mfa 로 다시 시도"
    return {"summary": summary}


def deploy(config: dict, args, who: dict) -> dict:
    """코드와 모델을 한 파일로 묶어 Lambda 함수를 만들거나 갱신한다."""
    cfg = config["cloud"]
    name = cfg["function_name"]
    version = model_version(common.MODEL_PATH.read_bytes())
    code = aws.build_zip()
    code_sha256 = base64.b64encode(hashlib.sha256(code).digest()).decode("ascii")
    # --model-path를 주면 함수가 읽을 위치만 바꾼다 (오류 재현용). 묶음 안의 모델 위치는 그대로다.
    configured_path = args.model_path or aws.PACKAGED_MODEL
    role_arn = aws.client("iam", config).get_role(RoleName=cfg["role_name"])["Role"]["Arn"]
    settings = {
        "Runtime": cfg["runtime"],
        "Role": role_arn,
        "Handler": aws.HANDLER,
        "Timeout": cfg["timeout_s"],
        "MemorySize": cfg["memory_mb"],
        "Description": f"SMS spam classifier (model {version})",
        "Environment": {"Variables": {
            "MODEL_PATH": configured_path,
            "MAX_CHARS": str(config["serve"]["max_chars"]),
        }},
    }
    lam = aws.client("lambda", config)
    if _function_config(lam, name) is None:
        lam.create_function(FunctionName=name, Code={"ZipFile": code},
                            Architectures=["x86_64"], Publish=False, **settings)
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
        action = "created"
    else:
        lam.update_function_code(FunctionName=name, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        lam.update_function_configuration(FunctionName=name, **settings)
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        action = "updated"

    # 로그 보관 기간을 정해 로그가 계속 쌓이지 않게 한다. 권한이 없으면 건너뛰고 기록한다.
    logs = aws.client("logs", config)
    group = aws.log_group(config)
    try:
        try:
            logs.create_log_group(logGroupName=group)
        except ClientError as exc:
            if aws.error_code(exc) != "ResourceAlreadyExistsException":
                raise
        logs.put_retention_policy(logGroupName=group, retentionInDays=cfg["log_retention_days"])
        log_retention = f"{cfg['log_retention_days']} days"
    except ClientError as exc:
        log_retention = f"not set ({aws.error_code(exc)})"

    info = lam.get_function_configuration(FunctionName=name)
    result = {
        "model_version": version,
        "inputs": {
            "model": common.rel(common.MODEL_PATH),
            "code": "spamsvc/",
            "package_sha256": code_sha256,
        },
        "outputs": {
            "function": info["FunctionArn"],
            "role": role_arn,
            "log_group": group,
        },
        "summary": {
            "action": action,
            "state": info.get("State"),
            "last_update_status": info.get("LastUpdateStatus"),
            "deployed_package_matches_local": info.get("CodeSha256") == code_sha256,
            "configured_model_path": configured_path,
            "model_path_is_valid": configured_path == aws.PACKAGED_MODEL,
            "log_retention": log_retention,
        },
    }
    if info.get("CodeSha256") != code_sha256:
        raise common.StepFailed("배포된 묶음이 로컬에서 만든 묶음과 다릅니다", result)
    return result


def invoke(config: dict, args, who: dict) -> dict:
    """배포된 함수에 문자 한 건을 보내고 응답을 기록한다."""
    name = config["cloud"]["function_name"]
    lam = aws.client("lambda", config)
    status, body, request_id = aws.invoke_function(lam, name, {"text": args.text})
    result = {
        "model_version": body.get("model_version"),
        "inputs": {"function": name, "text": args.text},
        "outputs": {"status": status, "response": body, "aws_request_id": request_id},
        "summary": {"status": status, "label": body.get("label"),
                    "spam_probability": body.get("spam_probability"), "error": body.get("error")},
    }
    if status != 200:
        raise common.StepFailed(f"함수가 {status}을 돌려줬습니다: {body.get('error')}", result)
    return result


def logs(config: dict, args, who: dict) -> dict:
    """최근 CloudWatch 로그를 가져와 파일로 저장한다 (오류 원인 조사용)."""
    group = aws.log_group(config)
    client = aws.client("logs", config)
    start = int((time.time() - args.minutes * 60) * 1000)
    events, token = [], None
    while True:
        extra = {"nextToken": token} if token else {}
        page = client.filter_log_events(logGroupName=group, startTime=start, **extra)
        events.extend(page.get("events", []))
        token = page.get("nextToken")
        if not token:
            break
    rows = common.mask(
        [{"timestamp": e["timestamp"], "message": e["message"].rstrip()} for e in events],
        who["hidden"])
    path = common.REPORT_DIR / "cloud_logs" / f"{args.run_id}.jsonl"
    common.write_jsonl(path, rows)
    failures = [r["message"] for r in rows if '"event": "failed"' in r["message"]]
    return {
        "inputs": {"log_group": group, "minutes": args.minutes},
        "outputs": {"log_file": common.rel(path)},
        "summary": {"events": len(rows), "failed_events": len(failures),
                    "last_failure": failures[-1] if failures else None},
    }


def status(config: dict, args, who: dict) -> dict:
    """함수와 로그 그룹의 현재 상태를 조회해 파일로 저장한다."""
    cfg = config["cloud"]
    name = cfg["function_name"]
    group = aws.log_group(config)

    info = _function_config(aws.client("lambda", config), name)
    function = {"name": name, "exists": info is not None}
    if info:
        function.update(
            state=info.get("State"),
            last_update_status=info.get("LastUpdateStatus"),
            last_modified=info.get("LastModified"),
            runtime=info.get("Runtime"),
            memory_mb=info.get("MemorySize"),
            timeout_s=info.get("Timeout"),
            package_sha256=info.get("CodeSha256"),
            package_bytes=info.get("CodeSize"),
            role=info.get("Role"),
            environment=info.get("Environment", {}).get("Variables", {}),
        )

    group_info = _log_group_info(aws.client("logs", config), group)
    log_state = {"log_group": group, "exists": group_info is not None}
    if group_info:
        log_state.update(retention_days=group_info.get("retentionInDays"),
                         stored_bytes=group_info.get("storedBytes"))

    report = {"checked_at": common.utc_now(), "region": cfg["region"],
              "function": function, "logs": log_state}
    path = common.REPORT_DIR / "cloud_status" / f"{args.run_id}.json"
    common.write_json(path, common.mask(report, who["hidden"]))
    return {
        "outputs": {"status_file": common.rel(path)},
        "summary": {
            "function_exists": function["exists"],
            "function_state": function.get("state"),
            "log_group_exists": log_state["exists"],
        },
    }


def destroy(config: dict, args, who: dict) -> dict:
    """이 프로젝트가 만든 함수와 로그 그룹을 지운다. --yes가 없으면 대상만 보여준다.

    실행 역할은 학교가 제공한 공용 자원이므로 건드리지 않는다.
    """
    name = config["cloud"]["function_name"]
    group = aws.log_group(config)
    targets = {"function": name, "log_group": group}
    if not args.yes:
        return {"summary": {"dry_run": True, "would_delete": targets}}

    deleted, missing = [], []

    lam = aws.client("lambda", config)
    if _function_config(lam, name) is None:
        missing.append("function")
    else:
        lam.delete_function(FunctionName=name)
        deleted.append("function")

    logs_client = aws.client("logs", config)
    if _log_group_info(logs_client, group) is None:
        missing.append("log_group")
    else:
        logs_client.delete_log_group(logGroupName=group)
        deleted.append("log_group")

    return {"summary": {"dry_run": False, "targets": targets,
                        "deleted": deleted, "not_found": missing}}


COMMANDS = {"check": check, "deploy": deploy, "invoke": invoke,
            "logs": logs, "status": status, "destroy": destroy}

"""AWS 접근 도구.

인증 정보는 boto3의 기본 방식(~/.aws/credentials 또는 환경변수)으로 읽는다.
코드와 기록에는 인증 정보, 계정 번호, 사용자 이름을 남기지 않는다.
"""
from __future__ import annotations

import io
import json
import zipfile

import boto3
from botocore.exceptions import ClientError

from pipeline import common

HANDLER = "spamsvc.handler.lambda_handler"
SERVICE_DIR = common.ROOT / "spamsvc"
PACKAGED_MODEL = "model/model.json"  # 배포 묶음 안에서의 모델 위치


def client(service: str, config: dict):
    return boto3.client(service, region_name=config["cloud"]["region"])


def error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def identity(config: dict) -> dict:
    """로그인한 사용자를 확인하고, 기록에서 가릴 값(hidden)을 함께 돌려준다.

    사용자 이름은 학번일 수 있어 계정 번호와 함께 가린다.
    """
    caller = client("sts", config).get_caller_identity()
    account, user = caller["Account"], caller["Arn"].split("/")[-1]
    return {"account": account, "user": user, "hidden": {account: "<account>", user: "<user>"}}


def resolve(config: dict, user: str) -> dict:
    """설정의 함수 이름에 있는 {user} 자리에 로그인한 사용자 이름을 넣는다.

    공용 계정에서 누구의 자원인지 드러나게 하되, 저장소에는 사용자 이름을 적지 않기 위해서다.
    """
    cloud = dict(config["cloud"])
    cloud["function_name"] = cloud["function_name"].format(user=user)
    return {**config, "cloud": cloud}


def log_group(config: dict) -> str:
    return f"/aws/lambda/{config['cloud']['function_name']}"


def build_zip() -> bytes:
    """서비스 코드(spamsvc)와 모델 파일을 담은 배포 묶음. 내용이 같으면 항상 같은 파일이 나온다.

    제공된 실행 역할에는 로그 쓰기 권한만 있어서, 모델을 S3에서 읽지 않고 묶음에 함께 넣는다.
    """
    files = {f"spamsvc/{path.name}": path.read_bytes() for path in sorted(SERVICE_DIR.glob("*.py"))}
    files[PACKAGED_MODEL] = common.MODEL_PATH.read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, content)
    return buffer.getvalue()


def invoke_function(lam, name: str, event: dict) -> tuple[int, dict, str | None]:
    """함수를 직접 호출하고 (상태 코드, 응답 본문, AWS 요청 ID)를 돌려준다."""
    response = lam.invoke(FunctionName=name, Payload=json.dumps(event).encode("utf-8"))
    request_id = response.get("ResponseMetadata", {}).get("RequestId")
    payload = json.loads(response["Payload"].read() or b"null")
    if response.get("FunctionError") or not isinstance(payload, dict) or "statusCode" not in payload:
        return 502, {"error": "function_error", "detail": payload}, request_id
    return payload["statusCode"], json.loads(payload["body"]), request_id

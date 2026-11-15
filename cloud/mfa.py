"""MFA 코드로 임시 인증 정보를 받아 ~/.aws/credentials의 [mlops-mfa] 프로필에 저장한다.

MFA 없이 만든 액세스 키로는 대부분의 작업이 거부되는 계정에서 쓴다.

    python -m cloud.mfa            # 6자리 코드를 물어본다

이후 명령은 환경변수 AWS_PROFILE=mlops-mfa 를 준 상태로 실행한다.
임시 인증 정보는 화면에 출력하지 않으며 12시간 뒤 만료된다.
"""
from __future__ import annotations

import configparser
import sys
from pathlib import Path

import boto3

PROFILE = "mlops-mfa"
CREDENTIALS = Path.home() / ".aws" / "credentials"


def main() -> int:
    session = boto3.Session(profile_name="default")  # MFA 전의 장기 액세스 키
    user = session.client("sts").get_caller_identity()["Arn"].split("/")[-1]
    devices = session.client("iam").list_mfa_devices(UserName=user)["MFADevices"]
    if not devices:
        print("등록된 MFA 기기가 없습니다. 콘솔의 '보안 자격 증명'에서 먼저 등록하세요.")
        return 1
    code = input("MFA 6자리 코드: ").strip()
    credentials = session.client("sts").get_session_token(
        SerialNumber=devices[0]["SerialNumber"], TokenCode=code)["Credentials"]

    parser = configparser.RawConfigParser()
    parser.read(CREDENTIALS, encoding="utf-8-sig")
    parser[PROFILE] = {
        "aws_access_key_id": credentials["AccessKeyId"],
        "aws_secret_access_key": credentials["SecretAccessKey"],
        "aws_session_token": credentials["SessionToken"],
    }
    with open(CREDENTIALS, "w", encoding="utf-8") as f:
        parser.write(f)
    print(f"[{PROFILE}] 프로필에 저장했습니다. 만료 시각: {credentials['Expiration']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

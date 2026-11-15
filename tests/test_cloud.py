"""배포 흐름 검사. 실제 AWS 대신 메모리 안의 가짜 Lambda/IAM/로그를 쓴다."""
import base64
import hashlib
import io
import json
import os
import shutil
import zipfile

import pytest
from botocore.exceptions import ClientError

import run
from cloud import aws, cli
from pipeline import common
from spamsvc import handler

ACCOUNT = "123456789012"
USER = "student-9999"
ROLE = common.load_config()["cloud"]["role_name"]


def client_error(code, message="not found"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "op")


class Waiter:
    def wait(self, **kwargs):
        pass


class FakeAws:
    def __init__(self, task_dir):
        self.task_dir = task_dir  # Lambda의 /var/task 역할
        self.function = None
        self.deny_writes = False
        self.roles = {ROLE}
        self.log_groups = {}
        self.log_events = []

    # sts, iam
    def get_caller_identity(self):
        return {"Account": ACCOUNT, "Arn": f"arn:aws:iam::{ACCOUNT}:user/{USER}"}

    def get_role(self, RoleName):
        if RoleName not in self.roles:
            raise client_error("NoSuchEntity")
        return {"Role": {"RoleName": RoleName, "Arn": f"arn:aws:iam::{ACCOUNT}:role/{RoleName}"}}

    # lambda
    def list_functions(self, **kwargs):
        return {"Functions": []}

    def _unpack(self, code):
        shutil.rmtree(self.task_dir, ignore_errors=True)
        zipfile.ZipFile(io.BytesIO(code)).extractall(self.task_dir)
        handler._cache.clear()  # 새 실행 환경

    def get_function_configuration(self, FunctionName):
        if self.function is None:
            raise client_error("ResourceNotFoundException")
        code = self.function["code"]
        return {**self.function["settings"], "State": "Active", "LastUpdateStatus": "Successful",
                "FunctionArn": f"arn:aws:lambda:ap-northeast-2:{ACCOUNT}:function:{FunctionName}",
                "CodeSha256": base64.b64encode(hashlib.sha256(code).digest()).decode(),
                "CodeSize": len(code)}

    def create_function(self, FunctionName, Code, **settings):
        if self.deny_writes:
            raise client_error("AccessDeniedException", (
                f"User: arn:aws:iam::{ACCOUNT}:user/{USER} is not authorized to perform: "
                "lambda:CreateFunction"))
        settings.pop("Architectures"), settings.pop("Publish")
        self.function = {"name": FunctionName, "code": Code["ZipFile"], "settings": settings}
        self._unpack(Code["ZipFile"])

    def update_function_code(self, FunctionName, ZipFile):
        self.function["code"] = ZipFile
        self._unpack(ZipFile)

    def update_function_configuration(self, FunctionName, **settings):
        self.function["settings"] = settings
        handler._cache.clear()

    def delete_function(self, FunctionName):
        self.function = None

    def get_waiter(self, name):
        return Waiter()

    def invoke(self, FunctionName, Payload):
        variables = self.function["settings"]["Environment"]["Variables"]
        saved = {k: os.environ.get(k) for k in variables}
        cwd = os.getcwd()
        os.environ.update(variables)
        os.chdir(self.task_dir)
        try:
            response = handler.lambda_handler(json.loads(Payload), None)
        finally:
            os.chdir(cwd)
            for key, value in saved.items():
                os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        self.log_events.append({"timestamp": 0, "message": json.dumps({"event": "invoked"})})
        return {"Payload": io.BytesIO(json.dumps(response).encode()),
                "ResponseMetadata": {"RequestId": "fake-request"}}

    # logs
    def describe_log_groups(self, **kwargs):
        prefix = kwargs.get("logGroupNamePrefix", "")
        return {"logGroups": [{"logGroupName": n, **v} for n, v in self.log_groups.items()
                              if n.startswith(prefix)]}

    def create_log_group(self, logGroupName):
        self.log_groups.setdefault(logGroupName, {})

    def put_retention_policy(self, logGroupName, retentionInDays):
        self.log_groups[logGroupName]["retentionInDays"] = retentionInDays

    def delete_log_group(self, logGroupName):
        del self.log_groups[logGroupName]

    def filter_log_events(self, **kwargs):
        return {"events": self.log_events}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    fake = FakeAws(tmp_path / "task")
    monkeypatch.setattr(aws.boto3, "client", lambda service, **kwargs: fake)
    monkeypatch.setattr(common, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(common, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    return fake


def records(step=None):
    rows = common.read_jsonl(common.RUN_LOG)
    return [r for r in rows if step is None or r["step"] == step]


def test_deploy_then_smoke_and_invoke(fake, capsys):
    assert cli.main(["check"]) == 0
    assert records("cloud:check")[-1]["summary"]["role"] == "ok"
    assert cli.main(["deploy"]) == 0
    assert fake.function["settings"]["Role"].endswith(":role/" + ROLE)
    first = records("cloud:deploy")[-1]["summary"]
    assert (first["action"], first["deployed_package_matches_local"]) == ("created", True)

    assert run.main(["--steps", "smoke", "--smoke-lambda"]) == 0
    assert records("smoke")[-1]["summary"] == {"passed": 8, "failed": 0, "failed_cases": []}

    assert cli.main(["invoke", "--text", "WINNER!! Call now to claim your cash prize"]) == 0
    assert records("cloud:invoke")[-1]["outputs"]["response"]["label"] == "spam"

    assert cli.main(["deploy"]) == 0
    assert records("cloud:deploy")[-1]["summary"]["action"] == "updated"


def test_wrong_model_path_fails_and_redeploy_recovers(fake, capsys):
    assert cli.main(["deploy", "--model-path", "model/missing.json"]) == 0
    assert records("cloud:deploy")[-1]["summary"]["model_path_is_valid"] is False

    assert cli.main(["invoke", "--text", "hello there"]) == 1
    failed = records("cloud:invoke")[-1]
    assert failed["success"] is False
    assert failed["outputs"]["response"]["error"] == "model_unavailable"
    assert run.main(["--steps", "smoke", "--smoke-lambda"]) == 1

    assert cli.main(["deploy"]) == 0
    assert cli.main(["invoke", "--text", "hello there"]) == 0
    assert run.main(["--steps", "smoke", "--smoke-lambda"]) == 0


def test_missing_role_stops_deploy_before_creating_anything(fake, capsys):
    fake.roles.clear()
    assert cli.main(["deploy"]) == 1
    assert records("cloud:deploy")[-1]["error"]["type"] == "ClientError"
    assert fake.function is None


def test_status_and_destroy(fake, capsys):
    cli.main(["deploy"])
    cli.main(["status"])
    assert records("cloud:status")[-1]["summary"] == {
        "function_exists": True, "function_state": "Active", "log_group_exists": True}

    cli.main(["destroy"])
    assert records("cloud:destroy")[-1]["summary"]["dry_run"] is True
    assert fake.function is not None

    assert cli.main(["destroy", "--yes"]) == 0
    assert records("cloud:destroy")[-1]["summary"]["deleted"] == ["function", "log_group"]
    assert fake.function is None and not fake.log_groups
    assert fake.roles == {ROLE}  # 학교가 제공한 역할은 그대로 둔다

    cli.main(["status"])
    assert records("cloud:status")[-1]["summary"] == {
        "function_exists": False, "function_state": None, "log_group_exists": False}
    cli.main(["destroy", "--yes"])
    assert records("cloud:destroy")[-1]["summary"]["not_found"] == ["function", "log_group"]


def test_logs_are_saved(fake, capsys):
    cli.main(["deploy"])
    cli.main(["invoke", "--text", "hello there"])
    assert cli.main(["logs"]) == 0
    assert records("cloud:logs")[-1]["summary"]["events"] == 1
    assert (common.REPORT_DIR / "cloud_logs").exists()


def test_account_number_and_user_name_never_reach_records(fake, capsys):
    cli.main(["deploy"])
    assert fake.function["name"] == f"{USER}_mlops"  # 실제 자원 이름에는 사용자 이름이 들어간다
    cli.main(["status"])
    cli.main(["invoke", "--text", "hello there"])
    cli.main(["logs"])
    run.main(["--steps", "smoke", "--smoke-lambda"])
    written = common.RUN_LOG.read_text(encoding="utf-8")
    written += "".join(p.read_text(encoding="utf-8") for p in common.REPORT_DIR.rglob("*.json*"))
    printed = capsys.readouterr().out
    for secret in (ACCOUNT, USER):
        assert secret not in written
        assert secret not in printed
    assert "<account>" in written and "<user>_mlops" in written


def test_denied_request_is_recorded_without_account_or_user_name(fake, capsys):
    fake.deny_writes = True
    assert cli.main(["deploy"]) == 1
    failed = records("cloud:deploy")[-1]
    assert failed["success"] is False
    assert "lambda:CreateFunction" in failed["error"]["message"]
    assert "user/<user>" in failed["error"]["message"]
    written = common.RUN_LOG.read_text(encoding="utf-8")
    assert ACCOUNT not in written and USER not in written


def test_deploy_package_is_stable_and_holds_service_code_and_model():
    first, second = aws.build_zip(), aws.build_zip()
    assert first == second
    package = zipfile.ZipFile(io.BytesIO(first))
    assert package.namelist() == ["spamsvc/__init__.py", "spamsvc/handler.py",
                                  "spamsvc/model.py", "model/model.json"]
    assert package.read("model/model.json") == common.MODEL_PATH.read_bytes()

"""AWS 명령 실행기. 모든 명령은 runs/run_log.jsonl에 기록된다.

    python -m cloud check                  # 인증 정보와 권한 확인 (자원을 만들지 않음)
    python -m cloud deploy                 # 코드와 모델을 묶어 Lambda 함수 생성/갱신
    python -m cloud invoke --text "..."    # 문자 한 건 예측
    python -m cloud logs --minutes 15      # 최근 로그 저장
    python -m cloud status                 # 자원 상태 조회
    python -m cloud destroy --yes          # 자원 삭제

배포된 함수의 전체 검사는 python run.py --steps smoke --smoke-lambda 로 한다.
"""
from __future__ import annotations

import argparse
import json
import sys

from pipeline import common

from . import aws
from .commands import COMMANDS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cloud", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    deploy = commands.add_parser("deploy")
    deploy.add_argument("--model-path", default=None,
                        help="함수가 읽을 모델 위치를 직접 지정 (오류 재현용)")
    invoke = commands.add_parser("invoke")
    invoke.add_argument("--text", required=True)
    logs = commands.add_parser("logs")
    logs.add_argument("--minutes", type=int, default=15)
    commands.add_parser("status")
    destroy = commands.add_parser("destroy")
    destroy.add_argument("--yes", action="store_true", help="실제로 삭제")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")
    config = common.load_config()
    args.run_id = common.new_run_id()
    hidden = {}

    def step() -> dict:
        who = aws.identity(config)
        hidden.update(who["hidden"])
        return COMMANDS[args.command](aws.resolve(config, who["user"]), args, who)

    def scrub(record: dict) -> dict:
        return common.mask(record, hidden)

    record = common.execute_step(args.run_id, f"cloud:{args.command}", step,
                                 config["cloud"], common.git_state(), scrub)
    if record["success"]:
        print(f"[cloud:{args.command}] ok ({record['duration_s']}s)")
    else:
        print(f"[cloud:{args.command}] FAILED: {record['error']['type']}: {record['error']['message']}")
    for key in ("summary", "outputs"):
        if record.get(key):
            print(json.dumps({key: record[key]}, ensure_ascii=False, indent=2))
    print(f"run_id={args.run_id} log={common.rel(common.RUN_LOG)}")
    return 0 if record["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

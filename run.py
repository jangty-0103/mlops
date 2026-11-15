"""단계별 실행: data -> train -> evaluate -> smoke.

각 단계의 결과를 runs/run_log.jsonl에 한 줄씩 남기고, 실패하면 이후 단계를 실행하지 않는다.

    python run.py                          # 전체 실행 (서비스 검사는 로컬 핸들러 대상)
    python run.py --steps evaluate smoke   # 일부 단계만
    python run.py --steps smoke --smoke-lambda   # 배포된 Lambda 함수를 검사
"""
from __future__ import annotations

import argparse
import json
import sys

from pipeline import common, data, evaluate, smoke, train

STEPS = {"data": data.run, "train": train.run, "evaluate": evaluate.run, "smoke": smoke.run}
CONFIG_SECTION = {"data": "data", "train": "train", "evaluate": "serve", "smoke": "serve"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--smoke-lambda", action="store_true", help="smoke 단계가 배포된 Lambda 함수를 검사")
    target.add_argument("--smoke-url", default=None, help="smoke 단계가 검사할 HTTP 주소")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")

    config = common.load_config()
    run_id = common.new_run_id()
    options = {"run_id": run_id, "smoke_url": args.smoke_url, "smoke_lambda": args.smoke_lambda,
               "hidden": {}}  # 단계가 채우는, 기록에서 가릴 값
    git = common.git_state()
    steps = [name for name in STEPS if name in args.steps]  # 항상 정해진 순서로 실행

    for position, name in enumerate(steps):
        record = common.execute_step(
            run_id, name, lambda: STEPS[name](config, options), config[CONFIG_SECTION[name]], git,
            scrub=lambda record: common.mask(record, options["hidden"]),
        )
        if record["success"]:
            print(f"[{name}] ok ({record['duration_s']}s) {json.dumps(record.get('summary', {}))}")
        else:
            print(f"[{name}] FAILED: {record['error']['type']}: {record['error']['message']}")
            skipped = steps[position + 1:]
            if skipped:
                print(f"skipped: {', '.join(skipped)}")
            print(f"run_id={run_id} log={common.rel(common.RUN_LOG)}")
            return 1

    print(f"run_id={run_id} log={common.rel(common.RUN_LOG)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

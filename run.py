"""단계별 실행: data -> train -> evaluate -> smoke.

각 단계의 결과를 runs/run_log.jsonl에 한 줄씩 남기고, 실패하면 이후 단계를 실행하지 않는다.

    python run.py                          # 전체 실행
    python run.py --steps evaluate smoke   # 일부 단계만
    python run.py --steps smoke --smoke-url https://...   # 배포된 서비스 검사
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

from pipeline import common, data, evaluate, smoke, train

STEPS = {"data": data.run, "train": train.run, "evaluate": evaluate.run, "smoke": smoke.run}
CONFIG_SECTION = {"data": "data", "train": "train", "evaluate": "serve", "smoke": "serve"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--smoke-url", default=None, help="smoke 단계가 검사할 배포 주소")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")

    config = common.load_config()
    options = {"smoke_url": args.smoke_url}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = common.git_state()
    environment = common.environment()
    steps = [name for name in STEPS if name in args.steps]  # 항상 정해진 순서로 실행

    for position, name in enumerate(steps):
        record = {
            "run_id": run_id,
            "step": name,
            "started_at": common.utc_now(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "environment": environment,
            "config": config[CONFIG_SECTION[name]],
            "model_version": None,
        }
        started = time.perf_counter()
        try:
            record.update(STEPS[name](config, options))
            record.update(success=True, error=None)
        except Exception as exc:
            record.update(getattr(exc, "result", {}))
            record.update(success=False, error={"type": type(exc).__name__, "message": str(exc)})
            if not isinstance(exc, common.StepFailed):
                traceback.print_exc()
        record["finished_at"] = common.utc_now()
        record["duration_s"] = round(time.perf_counter() - started, 2)
        common.append_jsonl(common.RUN_LOG, record)

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

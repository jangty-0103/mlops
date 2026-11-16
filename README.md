# 스팸 문자 분류 서비스

클라우드 MLOps 대체 실습과제로 제출하는 저장소입니다. 영어 문자 메시지(SMS)가 스팸인지 판별하는 모델을 학습하고, AWS Lambda에서 실행한 결과를 기록했습니다.

이 문서에 run_id와 함께 적은 실행은 모두 `runs/run_log.jsonl`에 기록이 있습니다. 확인하지 못한 내용은 [12. 확인하지 못한 것](#12-확인하지-못한-것)에 따로 적었습니다.

## 1. 서비스 목적과 구성

| 항목 | 내용 |
|---|---|
| 사용할 사람 | 받은 문자가 스팸인지 자동으로 걸러내려는 메시지 앱이나 고객센터 시스템의 개발자 |
| 목적 | 문자 한 건이 스팸인지 정상인지 판별 |
| 입력 | 문자 내용. `{"text": "..."}` 형식, 1~1000자 |
| 결과 | `label`(`spam` 또는 `ham`), `spam_probability`(0~1), `model_version`, `request_id` |

```
[로컬]  원본 데이터 → 정리·분할 → 학습 → 모델 파일(model.json) → 평가
                                              │
                                              ▼
[AWS]   배포 묶음(서비스 코드 + 모델 파일) → Lambda 함수 → CloudWatch 로그
                                              ▲
                              boto3 직접 호출 (문자 내용 → 판별 결과)
```

화면과 상시 서버는 두지 않았습니다. 다른 프로그램이 문자를 한 건씩 보내고 결과를 받는 용도여서, 요청이 올 때만 실행되는 함수 하나로 충분하다고 판단했습니다.

| 폴더·파일 | 내용 |
|---|---|
| `spamsvc/` | 서비스 코드. 순수 파이썬 예측(`model.py`)과 Lambda 핸들러(`handler.py`). 표준 라이브러리만 사용 |
| `pipeline/` | 데이터 준비, 학습, 평가, 서비스 검사 단계 |
| `run.py` | 위 단계를 순서대로 실행하고 기록 |
| `cloud/` | AWS 권한 확인, 배포, 호출, 로그 저장, 상태 조회, 자원 정리, MFA 인증 |
| `tests/` | 검사 코드 30개 |
| `config.json` | 데이터 주소와 해시, 학습 설정, 서비스 설정, 클라우드 설정 |
| `data/raw/` | 원본 데이터 |
| `artifacts/` | 학습된 모델과 단계별 결과 보고서 |
| `runs/run_log.jsonl` | 모든 실행의 기록 |

## 2. 데이터와 모델

### 출처와 사용 조건

- **데이터**: SMS Spam Collection, UCI Machine Learning Repository
  - 페이지: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
  - 인용: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
  - 사용 조건: CC BY 4.0입니다. 출처를 밝히면 공유와 변경이 허용되므로 원본 파일을 `data/raw/sms_spam_collection.zip`에 포함했습니다.
  - 확보 방법: `python run.py --steps data`를 실행하면 파일이 없을 때 위 주소에서 내려받고, `config.json`에 적힌 sha256과 다르면 중단합니다.
- **모델**: 사전학습 모델을 쓰지 않고 이 데이터로 직접 학습했습니다. 학습 코드는 `pipeline/train.py`, 결과는 `artifacts/model/model.json`입니다.

### 데이터 한 건의 의미

한 건은 문자 메시지 하나입니다. 원본 파일의 한 줄이 `정답<TAB>문자 내용` 형식입니다.

| 항목 | 내용 |
|---|---|
| 입력 | 문자 내용 (영어 문자열) |
| 정답 | `spam`(광고·사기 문자) 또는 `ham`(정상 문자) |
| 단위 | 건. 예측값은 스팸일 확률(0~1, 단위 없음) |

### 잘못된 값의 처리

`pipeline/data.py`에서 처리하고, 건수는 `artifacts/reports/data_report.json`에 남깁니다.

| 처리 | 기준 | 이번 데이터에서의 건수 |
|---|---|---|
| 형식이 틀린 줄 제거 | 탭이 없거나 정답이 `ham`/`spam`이 아님 | 0 |
| 빈 문자 제거 | 앞뒤 공백을 뺀 내용이 없음 | 0 |
| HTML 표기 복원 | `&lt;`, `&amp;` 등을 실제 문자로 되돌림 | - |
| 중복 제거 | 내용이 완전히 같은 문자는 첫 건만 남김 | 414 |
| 정답 충돌 제거 | 같은 내용인데 정답이 다르면 모두 제거 | 0 |

원본 5,574건 중 5,160건이 남았습니다(정상 4,518건, 스팸 642건).

### 학습용과 평가용 분리

- 중복을 먼저 제거한 뒤 나눠서, 같은 문자가 양쪽에 함께 들어가지 않게 했습니다. 겹치는 건수가 0인지 코드에서 확인합니다.
- 정상과 스팸의 비율을 유지하면서 무작위로 8:2로 나눴습니다(시드 42). 원본에 수신 시각이 없어서 시간 기준으로는 나눌 수 없었습니다.
- 학습용은 4,128건(정상 3,614, 스팸 514), 평가용은 1,032건(정상 904, 스팸 128)입니다.

### 예측할 때 알 수 없는 정보를 쓰지 않도록 한 것

- 입력은 문자 내용뿐이고, 정답 열은 입력에 넣지 않습니다.
- 단어 목록과 단어 가중치(TF-IDF)는 학습용 자료로만 만듭니다.
- 설정값 C는 학습용 자료 안의 5겹 교차검증으로만 골랐습니다. 평가용 자료는 학습 단계에서 읽지 않습니다.
- 판별 기준 확률(0.5)은 평가 결과를 보고 조정하지 않았습니다.

### 모델과 버전

| 항목 | 내용 |
|---|---|
| 방법 | TF-IDF(소문자 변환, 2글자 이상 단어, 2건 이상 나온 단어만) + 로지스틱 회귀(`class_weight=balanced`) |
| C 선택 | 0.1, 1, 10, 100 중 교차검증 F1이 가장 높은 10 (평균 0.933, 표준편차 0.028) |
| 단어 수 | 3,279 |
| 모델 파일 | `artifacts/model/model.json` (156,832바이트). 단어별 가중치와 절편을 담은 JSON |
| 모델 버전 | `94c479816748` (모델 파일 sha256의 앞 12자리) |

서비스는 scikit-learn 없이 `spamsvc/model.py`의 순수 파이썬 코드로 예측합니다. 학습 단계에서 이 계산이 scikit-learn의 확률과 같은지 학습용 4,128건 전체로 비교하며, 최대 차이는 2.2e-16이었습니다. 같은 자료와 설정으로 세 번 학습했고 세 번 모두 같은 모델 버전이 나왔습니다.

## 3. 결과 평가

### 평가 자료와 기준

- **자료**: 평가용 1,032건입니다. 학습과 설정 선택에는 쓰지 않았습니다.
- **판단 기준**: 스팸을 놓치는 것과 정상 문자를 스팸으로 막는 것이 모두 문제이므로, 스팸 기준의 정밀도, 재현율, F1을 봅니다. 정상 문자가 87.6%여서 정확도만 보면 스팸을 하나도 못 잡는 방법도 높게 나오기 때문입니다.
- **계산 방법**: 스팸 확률이 0.5 이상이면 스팸으로 판별합니다. 정밀도는 스팸으로 판별한 것 중 실제 스팸의 비율, 재현율은 실제 스팸 중 잡아낸 비율, F1은 둘의 조화평균입니다.
- **기준 방법**: 학습용 자료에서 가장 많은 종류인 "정상"을 항상 답하는 방법과 비교했습니다.
- **평가 대상**: 서비스가 실제로 읽는 `model.json`을 `spamsvc/model.py`로 불러서 평가합니다.

### 결과

| 지표 | 모델 | 기준 방법 |
|---|---|---|
| 정확도 | 0.9835 | 0.8760 |
| 스팸 정밀도 | 0.9302 | 0.0 |
| 스팸 재현율 | 0.9375 | 0.0 |
| 스팸 F1 | 0.9339 | 0.0 |

| 실제 \ 판별 | 스팸 | 정상 |
|---|---|---|
| 스팸 128건 | 120 | 8 |
| 정상 904건 | 9 | 895 |

원본은 `artifacts/reports/eval_report.json`입니다.

### 결과의 의미

- 기준 방법은 스팸을 한 건도 잡지 못합니다. 모델은 스팸 128건 중 120건을 잡았습니다.
- 정상 문자 904건 중 9건(약 1%)을 스팸으로 잘못 막았습니다. 정상 문자를 놓치면 안 되는 용도라면 판별 기준 확률을 높이거나 사람이 확인하는 단계가 필요합니다.
- 평가용 스팸이 128건뿐이어서 재현율은 한 건 차이로 0.78%p가 달라집니다. 재현율 93.75%의 오차 범위는 대략 ±4%p로 봐야 합니다.

### 잘못된 결과 한 건 분석

틀린 17건은 `artifacts/reports/eval_errors.jsonl`에 단어별 기여도와 함께 남겼습니다. 그중 한 건을 분석했습니다.

| 항목 | 내용 |
|---|---|
| 문자 | `Are you free now?can i call now?` |
| 정답 | 정상 |
| 판별 | 스팸 (확률 0.9227) |
| 점수를 올린 단어 | `call` +2.22, `now` +2.14, `free` +2.04 |
| 점수를 내린 단어 | `can` -1.08 |

- **원인**: 모델은 단어의 순서와 뜻을 보지 않고 어떤 단어가 있는지만 봅니다. `free`, `call`, `now`는 학습용 스팸("FREE prize, call now")에 자주 나와서 가중치가 큽니다. 이 문자에서 `free`는 "시간이 있다"는 뜻이지만 모델은 구분하지 못합니다.
- **문자 길이의 영향**: 단어가 6개뿐이어서 스팸 쪽 단어 3개의 비중이 커졌습니다.
- **같은 유형의 사례**: `Sir, i am waiting for your call, once free please call me.`도 같은 이유로 스팸으로 판별됐습니다(확률 0.6051).
- **개선 방향**: 두 단어 묶음(`free now`, `free prize`)을 특징에 넣으면 구분할 수 있을 것으로 보입니다. 이번 과제에서는 적용하지 않았습니다.

### 서비스 정상 동작 검사

모델 품질과는 별개로, 서비스가 입력을 받아 결과를 돌려주는지를 `pipeline/smoke.py`로 검사합니다. 로컬 핸들러와 배포된 Lambda 함수에 같은 8건을 보냅니다.

| 입력 | 기대 결과 |
|---|---|
| 전형적인 스팸 문자 | 200, `spam` |
| 전형적인 정상 문자 | 200, `ham` |
| 직접 호출 형식의 정상 문자 | 200 |
| `text` 누락 | 400 `missing_text` |
| 공백뿐인 `text` | 400 `empty_text` |
| 문자열이 아닌 `text` | 400 `invalid_text_type` |
| 1001자 | 400 `text_too_long` |
| JSON이 아닌 본문 | 400 `invalid_json` |

200 응답에 대해서는 `label` 값, 확률 범위, 모델 버전이 로컬 모델 파일과 같은지도 확인합니다. 로컬과 클라우드 모두 8건을 통과했습니다.

### 한계와 측정하지 못한 것

- 영어 문자만 다룹니다. 한국어 문자와 최근 유형의 스팸에 대한 성능은 측정하지 못했습니다. 데이터는 2012년에 공개된 영국·싱가포르 문자입니다.
- 판별 기준 확률 0.5가 최적인지는 검증하지 않았습니다.
- 응답 시간과 동시 요청 처리량은 측정하지 않았습니다. 로그에서 처음 실행 시 준비 시간 106ms, 처리 시간 2ms인 사례 한 건만 확인했습니다.
- 시간이 지나면서 성능이 떨어지는지 감시하는 기능과 자동 재학습은 없습니다.

## 4. 클라우드 실행

### 사용한 서비스와 설정

| 항목 | 값 |
|---|---|
| 서비스 | AWS Lambda (실행), CloudWatch Logs (로그) |
| 리전 | 서울 `ap-northeast-2` |
| 런타임 | `python3.12`, x86_64 |
| 메모리, 제한 시간 | 128MB, 10초 |
| 핸들러 | `spamsvc.handler.lambda_handler` |
| 환경변수 | `MODEL_PATH=model/model.json`, `MAX_CHARS=1000` |
| 실행 역할 | 학교에서 제공한 `yeonsung-lambda-basic-role` (로그 쓰기 권한만 있음) |
| 배포 묶음 | 서비스 코드 3개 파일 + 모델 파일, 55,414바이트 |
| 로그 보관 | 7일 |
| 호출 방식 | boto3 `lambda.invoke` 직접 호출. 공개 주소는 만들지 않음 |

설정값은 `config.json`의 `cloud` 항목에 있습니다. 함수 이름은 `{user}_mlops`로 적혀 있고, 실행할 때 AWS 로그인 이름이 `{user}` 자리에 들어갑니다. 로그인 이름이 학번이어서 저장소와 기록에는 `<user>`로 가려서 남겼습니다. 계정 번호도 `<account>`로 가렸습니다.

### 이 구성을 선택한 이유

- **Lambda**: 모델이 157KB짜리 파일 하나이고 예측 계산이 단순해서 서버를 계속 켜둘 필요가 없습니다. 요청이 없으면 비용이 0이고, 끄는 것을 잊어서 요금이 나갈 일도 없습니다.
- **모델을 배포 묶음에 포함**: 처음에는 모델을 S3에 두고 함수가 읽어오는 구조를 계획했습니다. 그런데 학교 계정에서는 실행 역할을 새로 만들 수 없고, 제공된 역할에는 S3 읽기 권한이 없었습니다. 그래서 모델을 코드와 함께 묶었습니다. 모델을 바꾸려면 다시 배포해야 하지만, 배포 기록에 코드와 모델 버전이 함께 남는 장점이 있습니다.
- **공개 주소 없음**: 인증 없는 주소를 열면 누구나 호출할 수 있어서 비용과 보안 위험이 생깁니다. 과제 검증에는 직접 호출로 충분하다고 판단했습니다.
- **서울 리전**: 학교 정책상 다른 리전은 사용할 수 없습니다.

### 데이터와 결과의 위치

| 대상 | 위치 |
|---|---|
| 모델 | Lambda 배포 묶음 안의 `model/model.json` (저장소의 `artifacts/model/model.json`과 같은 파일) |
| 호출 입력과 응답 | `runs/run_log.jsonl`의 `cloud:invoke` 기록, `artifacts/reports/cloud_smoke/` |
| 함수 실행 로그 | CloudWatch 로그 그룹 `/aws/lambda/<user>_mlops`, 사본은 `artifacts/reports/cloud_logs/` |
| 자원 상태 | `artifacts/reports/cloud_status/` |

함수 로그에는 문자 내용을 남기지 않고 글자 수만 남깁니다.

### 실제 입력과 결과

배포한 뒤 실제 문자를 보내서 받은 결과입니다. 시각은 모두 UTC이며, 한국 시각은 9시간을 더하면 됩니다.

| run_id | 입력 | 결과 |
|---|---|---|
| `20261115T154615Z-eb6dfc` | `Congratulations! You have won a free ticket. Text WIN to 80888 to claim your prize now.` | 200, `spam`, 확률 0.999755 |
| `20261115T154616Z-8be2d5` | `Are you coming home for dinner tonight?` | 200, `ham`, 확률 0.010941 |

두 응답 모두 `model_version`이 `94c479816748`로, 로컬에서 평가한 모델과 같습니다. 배포할 때 AWS에 올라간 묶음의 해시가 로컬에서 만든 묶음과 같은지도 확인합니다(`deployed_package_matches_local: true`).

| run_id | 단계 | 결과 |
|---|---|---|
| `20261115T154551Z-551e1f` | 배포 | 함수 생성, 상태 Active |
| `20261115T154613Z-abd7d0` | 서비스 검사 | 8건 통과 |
| `20261115T154617Z-02f17a` | 상태 조회 | 함수와 로그 그룹 존재 |

## 5. 실행 기록과 버전 관리

`run.py`와 `python -m cloud ...`의 모든 실행은 `runs/run_log.jsonl`에 한 줄씩 남습니다.

| 항목 | 내용 |
|---|---|
| `run_id`, `step` | 실행 번호와 단계 |
| `started_at`, `finished_at`, `duration_s` | 시각(UTC)과 걸린 시간 |
| `success`, `error` | 성공 여부와 오류 내용 |
| `git_commit`, `git_dirty` | 실행 시점의 커밋과, 커밋되지 않은 코드 변경이 있었는지 |
| `model_version` | 사용한 모델 버전 |
| `inputs` | 입력 파일과 해시, 호출 대상 |
| `config` | 그 단계에 쓰인 설정 |
| `outputs`, `summary` | 결과 파일 위치와 요약 |
| `environment` | Python, 운영체제, 라이브러리 버전 |

- **단계별 실행**: `python run.py`는 data → train → evaluate → smoke 순서로 실행합니다. 한 단계가 실패하면 기록을 남기고 이후 단계를 실행하지 않으며, 종료 코드 1로 끝납니다.
- **버전 연결**: 평가 단계는 모델에 기록된 학습 자료의 해시가 현재 학습 자료와 다르면 중단합니다. 클라우드 검사는 응답의 모델 버전이 로컬 모델과 다르면 실패로 처리합니다.
- **커밋 기록이 없는 초기 실행**: 맨 앞의 두 번(`20261115T143941Z-c71c3f`, `20261115T144006Z-cfe62c`)은 Git 저장소를 만들기 전이어서 `git_commit`이 비어 있습니다. 첫 커밋 뒤에 다시 실행한 기록이 `20261115T145225Z-a7b5c7`입니다.

## 6. 오류 재현과 복구

"잘못된 모델 경로" 오류를 클라우드에서 재현했습니다. 제 함수의 설정만 바꿨고, 공용 자원과 보안 설정은 건드리지 않았습니다. 전 과정을 커밋 `3c9be8d`에서 실행했습니다.

| 순서 | run_id | 한 일 | 결과 |
|---|---|---|---|
| 1 | `20261116T140844Z-5cc7c4` | `python -m cloud deploy --model-path model/missing.json` | 배포 성공. `model_path_is_valid: false` |
| 2 | `20261116T140850Z-032df3` | 스팸 문자 한 건 호출 | **실패**. 500 `model_unavailable`, 종료 코드 1 |
| 3 | `20261116T140852Z-dc66be` | 서비스 검사 | **실패**. 정상 입력 3건 실패, 잘못된 입력 5건은 통과 |
| 4 | `20261116T140914Z-6c776b` | `python -m cloud logs`로 로그 저장 | 원인 확인 |
| 5 | `20261116T140934Z-291d20` | `python -m cloud deploy`로 다시 배포 | 성공. `model_path_is_valid: true` |
| 6 | `20261116T140940Z-f18707` | 2번에서 실패한 요청 재실행 | 성공. 200, `spam`, 확률 0.998417 |
| 7 | `20261116T140942Z-ec965b` | 정상 입력 호출 | 성공. 200, `ham`, 확률 0.006531 |
| 8 | `20261116T140944Z-7100e7` | 서비스 검사 재실행 | 8건 통과 |
| 9 | `20261116T140946Z-dfaf05` | 상태 조회 | 함수 Active, `MODEL_PATH=model/model.json` |

- **실패한 요청**: `{"text": "URGENT! Your mobile number has won a 2000 pound prize. Call 09061790121 to claim."}`
- **오류 메시지**: 응답은 `{"error": "model_unavailable", "message": "모델을 불러오지 못했습니다"}`였습니다. 로그에는 `FileNotFoundError: [Errno 2] No such file or directory: 'model/missing.json'`이 남았습니다(`artifacts/reports/cloud_logs/20261116T140914Z-6c776b.jsonl`).
- **원인**: 함수의 환경변수 `MODEL_PATH`가 배포 묶음에 없는 파일을 가리켰습니다.
- **영향**: 모델이 필요한 요청은 모두 실패했습니다. 입력 검사에서 걸러지는 잘못된 입력은 모델을 읽기 전에 400으로 처리되어 영향이 없었습니다.
- **수정**: `MODEL_PATH`를 `model/model.json`으로 되돌려서 다시 배포했습니다.
- **실패와 정상 결과의 구분**: 실패한 응답에는 `label`이 없고 오류 코드만 있습니다. 기록에는 `success: false`로 남고, 명령은 종료 코드 1로 끝나서 이어지는 처리를 멈출 수 있습니다.

### 계획하지 않았던 실패

재현한 오류 외에, 진행하면서 실제로 겪은 문제도 기록에 남아 있습니다.

| run_id | 문제 | 원인과 조치 |
|---|---|---|
| `20261115T151546Z-cc8a05` | 권한 확인에서 Lambda와 로그 조회가 거부됨 | 학교 정책상 MFA 인증 없는 액세스 키의 요청은 거부됩니다. MFA 코드로 임시 인증을 받는 `cloud/mfa.py`를 만들었고, `20261115T152737Z-61b56f`에서 통과했습니다 |
| `20261116T140741Z-a4c8a5` 외 3건 | `ExpiredToken`으로 배포·호출·검사·로그 조회 실패 | 임시 인증의 유효 시간 12시간이 지났습니다. MFA 인증을 다시 받은 뒤 진행했습니다. 인증 단계에서 막혔기 때문에 AWS 자원은 바뀌지 않았습니다 |

## 7. 실행 방법

### 필요한 환경

- Python 3.12 (3.12.10에서 확인했습니다)
- Git
- 라이브러리와 버전은 `requirements.txt`에 있습니다. 서비스 코드(`spamsvc/`)는 표준 라이브러리만 사용합니다.

### 설치와 로컬 실행

```bash
git clone https://github.com/jangty-0103/mlops.git
cd mlops
python -m pip install -r requirements.txt
```

다른 프로젝트와 섞이지 않는 환경에서 실행하려면 마지막 줄 대신 가상환경을 만들어 설치합니다. 이 경우 이후 명령의 `python`을 `.\.venv\Scripts\python.exe`로 바꿔서 실행합니다(Windows 기준).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

검사 코드를 실행합니다. AWS 계정 없이 실행되며 30개가 통과해야 합니다.

```bash
python -m pytest -q
```

데이터 준비부터 서비스 검사까지 실행합니다. 원본 데이터가 저장소에 들어 있어서 내려받기 없이 실행됩니다.

```bash
python run.py
```

정상이라면 네 단계가 모두 `ok`로 나오고, 평가 수치와 모델 버전이 이 문서와 같아야 합니다.

```
[data] ok ... {"rows_after_cleaning": 5160, "train_rows": 4128, "test_rows": 1032}
[train] ok ... {"best_C": 10, "vocabulary_size": 3279, ...}
[evaluate] ok ... {"model_spam_f1": 0.9339, "baseline_spam_f1": 0.0, ...}
[smoke] ok ... {"passed": 8, "failed": 0, "failed_cases": []}
```

### 클라우드 실행에 필요한 권한

AWS 계정과 아래 권한이 필요합니다. 인증 정보는 `~/.aws/credentials`에 두며, 저장소에는 넣지 않습니다.

| 대상 | 필요한 권한 |
|---|---|
| Lambda 함수 `<로그인 이름>_mlops` | 생성, 코드·설정 변경, 조회, 호출, 삭제, 목록 조회 |
| 실행 역할 | 조회(`iam:GetRole`)와 함수에 연결(`iam:PassRole`). 역할에는 로그 쓰기 권한이 있어야 함 |
| CloudWatch 로그 그룹 `/aws/lambda/<로그인 이름>_mlops` | 생성, 보관 기간 설정, 조회, 삭제 |
| 사용자 계정 | `sts:GetCallerIdentity`, MFA를 쓰는 계정은 `iam:ListMFADevices`와 `sts:GetSessionToken` |

다른 계정에서 실행하려면 `config.json`의 `cloud.region`과 `cloud.role_name`을 그 계정에 맞게 바꿔야 합니다.

### 자원 준비와 실행

MFA가 필요한 계정은 먼저 임시 인증을 받습니다. 6자리 코드를 입력하면 `[mlops-mfa]` 프로필에 12시간짜리 인증 정보가 저장됩니다.

```bash
python -m cloud.mfa
```

이후 명령은 이 프로필을 쓰도록 환경변수를 설정한 터미널에서 실행합니다. 아래는 PowerShell 기준이며, bash에서는 `export AWS_PROFILE=mlops-mfa`를 사용합니다.

```powershell
$env:AWS_PROFILE = "mlops-mfa"
```

| 순서 | 명령 | 하는 일 |
|---|---|---|
| 1 | `python -m cloud check` | 인증과 조회 권한, 실행 역할 확인. 자원을 만들지 않음 |
| 2 | `python -m cloud deploy` | 코드와 모델을 묶어 함수 생성 또는 갱신 |
| 3 | `python run.py --steps smoke --smoke-lambda` | 배포된 함수에 8건을 보내 검사 |
| 4 | `python -m cloud invoke --text "문자 내용"` | 문자 한 건 판별 |
| 5 | `python -m cloud logs --minutes 15` | 최근 로그를 파일로 저장 |
| 6 | `python -m cloud status` | 함수와 로그 그룹 상태 조회 |
| 7 | `python -m cloud destroy --yes` | 함수와 로그 그룹 삭제. `--yes`가 없으면 대상만 보여줌 |

## 8. 새 폴더에서의 재실행

2026-11-16에 프로젝트 폴더와 다른 새 폴더(`C:\dev\mlops-rerun`)에 GitHub에서 저장소를 내려받고, 7절의 순서대로 실행했습니다.

| 항목 | 내용 |
|---|---|
| 내려받은 커밋 | `8c70d84` |
| 운영체제 | Windows 11 (10.0.26200) |
| Python | 3.12.10, 새로 만든 가상환경 |
| 설치된 버전 | scikit-learn 1.9.1, numpy 2.5.2, scipy 1.18.1, pytest 9.2.0, boto3 1.44.3 |

| 순서 | 명령 | run_id | 결과 |
|---|---|---|---|
| 1 | `git clone` | - | 파일 40개 |
| 2 | 가상환경 생성 후 `pip install -r requirements.txt` | - | 설치 성공 |
| 3 | `python -m pytest -q` | - | 30개 통과 |
| 4 | `python run.py` | `20261116T142515Z-07e2ec` | 4단계 성공. 모델 버전 `94c479816748`, F1 0.9339 |
| 5 | `python -m cloud check` | `20261116T142530Z-9c9469` | 통과 |
| 6 | `python -m cloud deploy` | `20261116T142532Z-0ecfb6` | 함수 갱신. 올라간 묶음이 로컬과 일치 |
| 7 | `python run.py --steps smoke --smoke-lambda` | `20261116T142539Z-06d681` | 8건 통과 |
| 8 | `python -m cloud invoke --text "Free entry in a weekly competition to win tickets. Text WIN to 87121 now."` | `20261116T142541Z-41f9e9` | 200, `spam`, 확률 0.972045 |
| 9 | `python -m cloud status` | `20261116T142541Z-e5177b` | 함수 Active |

- **같은 결과가 나왔는지**: 재실행 후 `git status`에서 바뀐 파일은 `runs/run_log.jsonl`과 `artifacts/reports/smoke_report_local.json` 두 개뿐이었습니다. 뒤의 파일은 검사 시각과 요청 번호가 들어가서 실행할 때마다 달라집니다. 모델 파일과 데이터·학습·평가 보고서는 저장소에 있던 것과 완전히 같았습니다.
- **기록**: 위 기록 9건은 모두 `git_commit`이 `8c70d84`이고 `git_dirty`가 `false`입니다.
- **범위**: 같은 PC의 새 폴더와 새 가상환경에서 확인했습니다. 다른 PC나 다른 운영체제에서는 확인하지 못했습니다.

## 9. 권한, 비용, 자원 정리

### 접근 권한과 대상

- 학교에서 발급받은 계정의 개인 사용자로 접근했습니다. 여러 학생이 같은 계정을 함께 쓰기 때문에, 이름에 제 로그인 이름이 들어간 함수와 그 로그 그룹만 만들고 지웁니다.
- 실행 역할은 학교에서 제공한 공용 자원이어서 수정하거나 삭제하지 않습니다. `destroy` 명령도 역할은 건드리지 않습니다.
- 학교 정책상 MFA 인증이 필요하고, 서울 리전만 사용할 수 있습니다.
- 액세스 키와 임시 인증 정보는 PC의 `~/.aws/credentials`에만 있습니다.

### 비용이 발생하는 자원과 제한 방법

| 자원 | 비용이 생기는 경우 | 제한 방법 |
|---|---|---|
| Lambda 함수 | 호출 횟수와 실행 시간. 호출하지 않으면 0 | 공개 주소를 만들지 않아 본인만 호출 가능. 메모리 128MB, 제한 시간 10초 |
| CloudWatch 로그 | 저장 용량 | 보관 기간 7일. 문자 내용을 로그에 남기지 않음 |

이 과제에서 함수를 호출한 횟수는 모두 38건입니다. 삭제 직전에 저장한 로그의 28건을 기준으로 과금 시간은 합계 0.705초, 한 건 최대 0.254초였고, 메모리는 최대 41MB를 사용했습니다. AWS 프리티어(월 100만 건, 40만 GB-초) 범위 안입니다. 삭제 직전에 조회한 로그 저장량은 2,441바이트였습니다.

### 자원 정리

2026-11-16에 이 과제로 만든 자원을 모두 삭제하고 상태를 다시 조회했습니다.

| 순서 | run_id | 한 일 | 결과 |
|---|---|---|---|
| 1 | `20261116T142717Z-d89dff` | 삭제 전 로그 저장 | 로그 116줄 저장 |
| 2 | `20261116T142718Z-459fc6` | 삭제 전 상태 조회 | 함수 Active, 로그 그룹 존재 |
| 3 | `20261116T142720Z-1f3ef4` | `python -m cloud destroy`로 삭제 대상 확인 | 함수와 로그 그룹 2개 |
| 4 | `20261116T142721Z-025c22` | `python -m cloud destroy --yes` | 함수와 로그 그룹 삭제 |
| 5 | `20261116T142722Z-c09952` | 삭제 후 상태 조회 | 함수 없음, 로그 그룹 없음 |
| 6 | `20261116T142724Z-38dcea` | 삭제 후 호출 시도 | 실패. `ResourceNotFoundException` (함수가 없음을 확인) |
| 7 | `20261116T142800Z-7efea6` | 38초 뒤 상태 재조회 | 함수 없음, 로그 그룹 없음 |

삭제 후의 상태는 `artifacts/reports/cloud_status/20261116T142800Z-7efea6.json`에 있습니다.

| 구분 | 내용 | 이유와 계획 |
|---|---|---|
| AWS에 남은 자원 | 이 과제로 만든 자원은 없습니다 | 실행 역할은 학교에서 제공한 공용 자원이어서 그대로 두었습니다 |
| 보관한 자료 | 로그 사본, 상태 조회 결과, 실행 기록 (저장소 안) | CloudWatch 로그는 로그 그룹과 함께 지워지므로, 실행 증빙으로 사본을 남겼습니다 |
| PC에 남은 것 | `~/.aws/credentials`의 액세스 키와 임시 인증 정보 | 임시 인증은 발급 12시간 뒤 만료됩니다. 액세스 키는 과제 제출 후 콘솔에서 삭제할 계획입니다 |

함수가 다시 필요하면 `python -m cloud deploy`로 만들 수 있습니다.

## 10. 결과 파일 위치

| 파일 | 내용 |
|---|---|
| `runs/run_log.jsonl` | 모든 실행의 기록 |
| `artifacts/model/model.json` | 학습된 모델 |
| `artifacts/reports/data_report.json` | 데이터 정리와 분할 결과 |
| `artifacts/reports/train_report.json` | 교차검증 결과, 모델 버전, 계산 일치 확인 |
| `artifacts/reports/eval_report.json` | 평가 지표와 기준 방법 비교 |
| `artifacts/reports/eval_errors.jsonl` | 틀린 17건과 단어별 기여도 |
| `artifacts/reports/smoke_report_local.json` | 로컬 서비스 검사 결과 |
| `artifacts/reports/cloud_smoke/` | 클라우드 서비스 검사 결과 (실행마다 한 파일) |
| `artifacts/reports/cloud_logs/` | 저장한 CloudWatch 로그 |
| `artifacts/reports/cloud_status/` | 자원 상태 조회 결과 |

`data/processed/`의 학습용·평가용 파일은 `python run.py --steps data`로 다시 만들 수 있어서 저장소에 넣지 않았습니다.

## 11. 참고 자료와 구현 범위

### 참고 자료

- 데이터: 2절의 SMS Spam Collection
- Almeida, T. A., Hidalgo, J. M. G., Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering.
- scikit-learn 문서: `TfidfVectorizer`, `LogisticRegression`, `GridSearchCV`
- AWS 문서: Lambda Python 런타임, boto3 Lambda·CloudWatch Logs·STS

### 구현 범위

| 구분 | 내용 |
|---|---|
| 이 저장소에서 작성한 부분 | 데이터 정리와 분할(`pipeline/data.py`), 순수 파이썬 예측 계산과 모델 파일 형식(`spamsvc/model.py`), Lambda 핸들러와 입력 검사(`spamsvc/handler.py`), 평가와 기준 방법 비교(`pipeline/evaluate.py`), 서비스 검사(`pipeline/smoke.py`), 실행 기록(`pipeline/common.py`, `run.py`), AWS 배포·정리 도구(`cloud/`), 검사 코드(`tests/`) |
| 라이브러리를 사용한 부분 | TF-IDF 계산, 로지스틱 회귀 학습, 교차검증은 scikit-learn을 사용했습니다. AWS 호출은 boto3를 사용했습니다 |
| 외부 코드 | 복사해 온 코드는 없습니다 |

## 12. 확인하지 못한 것

- 다른 PC나 다른 운영체제에서의 재실행: 같은 PC의 새 폴더와 새 가상환경에서만 확인했습니다(8절).
- HTTP 주소를 통한 호출: `run.py --smoke-url` 기능은 있지만, 공개 주소를 만들지 않아서 실제 주소로는 검증하지 못했습니다.
- 응답 시간과 동시 요청 처리량
- 한국어 문자와 최근 스팸에 대한 성능
- 외부 모델은 사용하지 않았으므로, 버전을 확인할 수 없는 범위는 없습니다.

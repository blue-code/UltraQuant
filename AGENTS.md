# Repository Guidelines

## 프로젝트 구조 및 모듈 구성
이 저장소는 루트 중심의 Python 프로젝트다. 핵심 실행/엔진 코드는 `ultra_quant.py`, 전략·분석 로직은 `strategy.py`, 테스트 코드는 `tests/test_core.py`에 있다. 실행 스크립트는 `run.bat`(Windows), `run.sh`(Unix)로 제공된다. 신규 모듈은 루트에 무분별하게 추가하지 말고, 기능 단위로 분리할 때는 `tests/`에 동일한 책임의 테스트를 함께 추가한다.

## 빌드/테스트/개발 명령
- `python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows): 로컬 개발 환경 생성/활성화
- `pip install -r requirements.txt`: 기본 의존성 설치
- `python ultra_quant.py`: 메인 엔진 단독 실행
- `python strategy.py`: 전략 분석 스크립트 실행
- `python -m unittest discover -s tests -p "test_*.py"`: 전체 테스트 실행
- `run.bat` 또는 `bash run.sh`: 의존성 설치 후 실행까지 일괄 처리

## 코딩 스타일 및 네이밍 규칙
Python은 PEP 8을 기본으로 하며, 들여쓰기는 스페이스 4칸을 사용한다. 클래스는 `PascalCase`, 함수/변수는 `snake_case`, 상수는 `UPPER_SNAKE_CASE`를 따른다. 타입 힌트와 `dataclass` 사용 패턴이 이미 존재하므로 신규 코드도 일관되게 유지한다. 복잡한 수치/시뮬레이션 로직은 짧은 주석으로 “이유”만 설명하고, 동작 자체를 설명하는 과한 주석은 지양한다.

## 테스트 가이드라인
테스트 프레임워크는 표준 `unittest`다. 테스트 파일명은 `test_*.py`, 테스트 메서드는 `test_*` 패턴을 사용한다. 신규 기능에는 정상 경로 1개 이상, 경계/예외 경로 1개 이상을 포함한다. 랜덤 데이터 기반 테스트는 재현 가능성을 위해 시드 고정(`np.random.seed(...)`)을 권장한다.

## 커밋 및 PR 가이드라인
최근 이력 기준 커밋 접두어는 `feat:`, `fix:`, `refactor:`, `docs:`를 사용한다. 커밋 메시지는 한글로 “무엇을 왜 바꿨는지”를 1줄로 명확히 작성한다. PR에는 변경 목적, 핵심 변경점, 테스트 결과(실행 명령 포함), 영향 범위를 반드시 포함한다.

## 보안 및 설정 팁
민감 정보는 코드에 하드코딩하지 않는다. 설정값은 예시 템플릿(`.env.example`)을 기준으로 분리하고, API 키/시크릿은 런타임 주입 원칙을 따른다.

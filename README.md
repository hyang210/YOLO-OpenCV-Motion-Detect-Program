# YOLO-OpenCV-Motion-Detect-Program

지하철 개찰구 영상에서 **뛰어넘기(Jump)** 와 **기어서 통과하기(Crawl)** 같은 무단 통과 의심 동작을 탐지하고 로그로 남기는 프로그램입니다.

YOLO 하나로 모든 사람을 가볍게 1차 판정하고, 판정이 애매한 후보에게만 MediaPipe를 써서 연산 비용을 낮췄습니다.

---

## 동작 방식

```
영상 프레임
   │
   ▼
[1] YOLO11-pose + ByteTrack ─────────── 모든 사람, 매 프레임 1회
   │  박스 · 추적 ID · keypoint 17개
   ▼
[2] ID별 베이스라인 수집 ────────────── 처음 1초 동안의 평상시 무릎/엉덩이 각도
   │
   ▼
[3] 1차 판정 (느슨한 기준) ───────────── 기준값의 70% / 발이 무릎 근처
   │  대부분의 사람은 여기서 Normal 로 종료
   ▼
[4] 최종 판정 (원래 기준)
   │  · 각도 변화 → YOLO 값으로 비교 (베이스라인과 같은 모델)
   │  · 발-무릎 높이 → 발이 무릎 근처인 후보만 MediaPipe 발끝/뒤꿈치로 재확인
   ▼
[5] 연속 확정 + 쿨다운 ──────────────── 0.1초 연속이면 확정, 같은 이벤트는 3초간 재기록 X
   │
   ▼
event_log.txt 기록 · 화면 표시
```

### 왜 두 모델을 나눠 쓰나

| | YOLO11-pose | MediaPipe Pose |
|---|---|---|
| 처리 단위 | 한 번에 여러 명 | 한 번에 한 명 (crop 필요) |
| keypoint | 17개 (발목까지) | 33개 (**발끝·뒤꿈치 포함**) |
| 비용 | 사람 수와 무관 | 사람 수에 비례 |
| 역할 | 추적, 1차 판정, 각도 판정 | 발이 무릎을 넘었는지 정밀 확인 |

MediaPipe는 호출 시점에만 로드되고 프레임당 최대 3회(`MAX_MP_PER_FRAME`)로 제한됩니다. 의심 동작이 없는 영상에서는 MediaPipe가 아예 로드되지 않습니다.

---

## 판정 기준

| 동작 | 조건 |
|---|---|
| **Jump** | 한쪽 발이 반대쪽 무릎보다 위 (박스 높이의 2% 이상) |
| **Jump** | 무릎 각도가 베이스라인 대비 80° 이상 변화 **그리고** 손목이 엉덩이보다 위 |
| **Crawl** | 무릎 각도 30° 이상 변화 **그리고** 엉덩이 각도 10° 이상 변화 **그리고** 상체가 수직에서 60° 이상 기울어짐 |

- 각도는 좌우 중 keypoint가 더 잘 보이는 쪽으로 계산합니다.
- 각도 기준은 사람별 베이스라인(평상시 자세)과의 **차이**입니다. 그래서 체형이나 카메라 각도의 영향을 덜 받습니다.
- 모든 값은 [`detector/config.py`](detector/config.py)에서 조정할 수 있습니다.

---

## 설치

Python 3.9 이상을 권장합니다.

```bash
pip install -r requirements.txt
```

YOLO 모델 파일(`yolo11n-pose.pt` 등)은 처음 실행할 때 자동으로 다운로드됩니다.

> **MediaPipe 버전 주의**: 이 코드는 `mediapipe.solutions.pose` (legacy API)를 사용합니다. 설치된 버전에 이 API가 없으면 실행 중 안내 메시지와 함께 종료되니, `mp.solutions`를 지원하는 버전으로 설치해 주세요.

---

## 실행

```bash
# 영상 파일
python main_logic.py --source sample.mp4 --skeleton

# 웹캠
python main_logic.py --source 0

# 화면 없이 로그만
python main_logic.py --source sample.mp4 --no-show
```

재생 중 `q`를 누르면 종료됩니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `sample.mp4` | 영상 경로 또는 웹캠 번호 |
| `--model` | `yolo11n-pose.pt` | YOLO pose 모델. 정확도가 부족하면 `yolo11s-pose.pt`, `yolo11m-pose.pt` |
| `--imgsz` | `640` | YOLO 입력 크기. 낮출수록 빠르지만 작은 사람을 놓칠 수 있음 |
| `--mp-complexity` | `1` | MediaPipe 모델 크기 (0 = 가장 가벼움, 2 = 가장 정확) |
| `--blur` | 끔 | 사람 외 배경 블러 |
| `--skeleton` | 끔 | YOLO 스켈레톤 표시 |
| `--no-show` | 끔 | 화면 출력 없이 실행 |
| `--log` | `event_log.txt` | 이벤트 로그 파일 경로 |

---

## 출력

### 화면
- 초록 박스: Normal / 빨간 박스: Jump / 파란 박스: Crawl
- 좌측 상단: 처리 FPS(화면 그리기 제외), 누적 MediaPipe 호출 수

### 로그 (`event_log.txt`)
```
[2026-10-05 14:03:21] Frame 412 | Person 7: Jump - Knee=96.3, Hip=141.0
```

### 종료 시 요약 (수치는 예시)
```
완료: 1800 프레임, 평균 24.5 FPS, MediaPipe 호출 37회 (0.02/프레임). 로그는 event_log.txt에 저장.
```
MediaPipe 호출이 프레임당 0에 가까울수록 의도대로 가볍게 동작하고 있다는 뜻입니다.

---

## 설정값 튜닝

[`detector/config.py`](detector/config.py)

| 값 | 기본 | 이렇게 조정 |
|---|---|---|
| `CONFIRM_SEC` | 0.1초 | 오탐이 많으면 늘리기 |
| `COOLDOWN_SEC` | 3초 | 같은 사람 로그가 너무 많으면 늘리기 |
| `SCREEN_RATIO` | 0.7 | 놓치는 동작이 많으면 낮추기 (후보 증가, 비용 증가) |
| `FOOT_MARGIN` | 0.02 | 발 관련 Jump 오탐이 많으면 늘리기 |
| `MAX_MP_PER_FRAME` | 3 | 사람이 많은 장면에서 정밀도가 부족하면 늘리기 (비용 증가) |
| `BASELINE_SEC` | 1초 | 베이스라인 수집 시간 |

시간 단위 값(`*_SEC`)은 영상 FPS에 맞춰 자동으로 프레임 수로 변환됩니다.

---

## 프로젝트 구조

```
main_logic.py              실행 진입점, 메인 루프
detector/
  config.py                임계값·시간 설정
  features.py              keypoint → 각도·발 높이 특징 계산, 동작 판정
  person_state.py          ID별 베이스라인, 연속 확정, 쿨다운
  confirmer.py             MediaPipe 후보 확인 (다른 사람 오검출 필터 포함)
  visualize.py             화면 그리기
requirements.txt
```

---

## 알려진 한계

- **2D 각도 기반**: 카메라를 정면으로 보고 있는 사람은 무릎 굽힘이 작게 보일 수 있습니다.
- **오탐 가능 자세**: 다리를 꼬고 앉기, 신발 끈 묶기, 물건 줍기 등은 Jump/Crawl 조건에 걸릴 수 있습니다.
- **판정 영역 제한 없음**: 현재 화면 전체에서 판정합니다. 개찰구가 아닌 곳의 동작도 탐지됩니다.
- **베이스라인**: 처음 등장할 때 이미 비정상 자세라면 베이스라인이 틀어질 수 있습니다.

## 개선 예정

- [ ] 판정 로직 단위 테스트
- [ ] `--stride` 옵션 (N프레임마다 처리)
- [ ] 개찰구 영역(ROI) 지정
- [ ] 이벤트 발생 시 프레임 캡처 저장

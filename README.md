# YOLO-OpenCV-Motion-Detect-Program

지하철 개찰구 등에서 사람의 **Jump(뛰어넘기) / Crawl(기어가기)** 동작을 탐지합니다.

- **YOLO11-pose + ByteTrack**: 모든 사람의 박스·ID·keypoint를 매 프레임 한 번에 추출하고 1차 판정
- **MediaPipe Pose**: 발이 무릎 근처까지 올라온 후보만 발끝/뒤꿈치로 발-무릎 높이를 정밀 확인
  (각도 변화 판정은 베이스라인과 같은 YOLO 값으로만 비교)

## 실행

```bash
pip install -r requirements.txt
python main_logic.py --source sample.mp4 --skeleton
```

| 옵션 | 설명 |
|---|---|
| `--source` | 영상 경로 또는 웹캠 번호 (기본 `sample.mp4`) |
| `--model` | YOLO pose 모델 (기본 `yolo11n-pose.pt`, 더 정확하게는 `yolo11s-pose.pt`) |
| `--mp-complexity` | MediaPipe 모델 크기 0/1/2 (기본 1) |
| `--blur` | 사람 외 배경 블러 |
| `--skeleton` | YOLO 스켈레톤 표시 |
| `--no-show` | 화면 출력 없이 실행 |
| `--log` | 이벤트 로그 파일 (기본 `event_log.txt`) |

## 구조

```
main_logic.py            실행 진입점, 메인 루프
detector/config.py       임계값·시간 설정
detector/features.py     keypoint → 각도 특징 계산, 동작 판정
detector/person_state.py ID별 베이스라인·연속 확정·쿨다운
detector/confirmer.py    MediaPipe 후보 확인
detector/visualize.py    화면 그리기
```

"""탐지 임계값 및 설정값"""
from dataclasses import dataclass

# YOLO 임계값
CONF_THRES = 0.5
NMS_THRES = 0.4
KP_CONF = 0.5              # keypoint 신뢰도(YOLO conf / MediaPipe visibility) 최소값

# 동작 판정 기준값 (베이스라인 대비 각도 변화량, 단위: 도)
THETA = 80.0               # Jump: 무릎 각도 변화
CRAWL_KNEE = 30.0          # Crawl: 무릎 각도 변화
CRAWL_HIP = 10.0           # Crawl: 엉덩이 각도 변화
CRAWL_TORSO = 60.0         # Crawl: 상체가 수직에서 기울어진 각도
FOOT_MARGIN = 0.02         # 발이 반대쪽 무릎보다 박스 높이의 몇 % 이상 위에 있어야 하는지

# 단계적 판정 설정
SCREEN_RATIO = 0.7         # 1차(YOLO) 판정은 기준값의 70% 로 느슨하게 → 후보를 넉넉히 뽑음
FOOT_SCREEN_MARGIN = -0.05 # 1차: 발이 반대쪽 무릎보다 박스 높이 5% 아래까지 올라오면 후보 → MediaPipe 로 확인
MAX_MP_PER_FRAME = 3       # 프레임당 MediaPipe 호출 상한 (비용 상한)
MIN_MP_BOX_H = 80          # 이보다 작은 박스는 MediaPipe 가 부정확하므로 YOLO 결과로 판정
CROP_PAD = 0.2

# 시간 기준(초). 영상 FPS 에 맞춰 프레임 수로 변환해서 사용
BASELINE_SEC = 1.0         # ID 별 베이스라인 수집 시간
CONFIRM_SEC = 0.1          # 이 시간 동안 연속 확인되어야 이벤트 확정
COOLDOWN_SEC = 3.0         # 같은 ID, 같은 이벤트 재기록 금지 시간
STALE_SEC = 2.0            # 이 시간 동안 안 보인 ID 는 상태 삭제
DEFAULT_FPS = 30.0         # 웹캠 등 FPS 를 알 수 없을 때


@dataclass(frozen=True)
class FrameParams:
    baseline: int
    confirm: int
    cooldown: int
    stale: int

    @classmethod
    def from_fps(cls, fps):
        if not fps or fps < 1 or fps > 240:
            fps = DEFAULT_FPS
        to_frames = lambda sec: max(1, round(sec * fps))
        return cls(to_frames(BASELINE_SEC), to_frames(CONFIRM_SEC),
                   to_frames(COOLDOWN_SEC), to_frames(STALE_SEC))

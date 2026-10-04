"""의심 후보만 MediaPipe 로 정밀 확인"""
import cv2

from .config import CROP_PAD
from .features import MP_KP


class MediaPipeConfirmer:
    """후보가 처음 생길 때 mediapipe 를 import/로드. 단일 인스턴스를 메인 스레드에서만 사용."""

    def __init__(self, complexity):
        self.complexity = complexity
        self._pose = None
        self.calls = 0

    def _load(self):
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            raise RuntimeError(
                f'설치된 mediapipe {mp.__version__} 에 legacy solutions API 가 없습니다. '
                'mp.solutions 를 지원하는 버전으로 설치해 주세요.')
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,       # 후보일 때만 띄엄띄엄 호출하므로 추적 모드 X
            model_complexity=self.complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def landmarks(self, frame, box):
        """box 주변 crop 에서 포즈 추정. 실패하거나 다른 사람을 잡았으면 None."""
        if self._pose is None:
            self._load()
        x1, y1, x2, y2 = box
        dx, dy = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
        xA, yA = max(0, x1 - dx), max(0, y1 - dy)
        xB, yB = min(frame.shape[1], x2 + dx), min(frame.shape[0], y2 + dy)
        if xB <= xA or yB <= yA:
            return None

        crop = frame[yA:yB, xA:xB]
        self.calls += 1
        res = self._pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark
        cw, ch = xB - xA, yB - yA
        pts = {n: (xA + lm[i].x * cw, yA + lm[i].y * ch) for n, i in MP_KP.items()}
        conf = {n: lm[i].visibility for n, i in MP_KP.items()}

        # 붐비는 곳에선 crop 안의 옆 사람을 잡을 수 있음 → 골반 중심이 YOLO 박스 밖이면 버림
        hx = (pts['l_hip'][0] + pts['r_hip'][0]) / 2
        hy = (pts['l_hip'][1] + pts['r_hip'][1]) / 2
        if not (x1 <= hx <= x2 and y1 <= hy <= y2):
            return None
        return pts, conf

    def close(self):
        if self._pose is not None:
            self._pose.close()

"""keypoint → 자세 특징 계산 및 동작 판정 (YOLO, MediaPipe 공통)"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import (CRAWL_HIP, CRAWL_KNEE, CRAWL_TORSO, FOOT_MARGIN, KP_CONF, THETA)

# keypoint 인덱스 (YOLO: COCO 17점, MediaPipe: 33점)
YOLO_KP = {
    'l_shoulder': 5, 'r_shoulder': 6, 'l_wrist': 9, 'r_wrist': 10,
    'l_hip': 11, 'r_hip': 12, 'l_knee': 13, 'r_knee': 14, 'l_ankle': 15, 'r_ankle': 16,
}
MP_KP = {
    'l_shoulder': 11, 'r_shoulder': 12, 'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24, 'l_knee': 25, 'r_knee': 26, 'l_ankle': 27, 'r_ankle': 28,
    'l_heel': 29, 'r_heel': 30, 'l_foot': 31, 'r_foot': 32,
}


@dataclass
class Features:
    knee: float | None          # 무릎 각도 (엉덩이-무릎-발목)
    hip: float | None           # 엉덩이 각도 (어깨-엉덩이-무릎)
    torso: float | None         # 상체가 수직에서 기울어진 각도
    wrist_above_hip: bool
    foot_gap: float | None      # (반대쪽 무릎 y - 발 최상단 y) / 박스 높이. 양수면 발이 무릎보다 위

    @property
    def foot_above_knee(self):
        return self.foot_gap is not None and self.foot_gap >= FOOT_MARGIN


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / mag
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def extract_features(pts, conf, box_h):
    """pts/conf: {이름: (x, y)} / {이름: 신뢰도}"""
    ok = lambda name: name in pts and conf[name] >= KP_CONF

    # 좌우 중 더 잘 보이는 쪽으로 각도 계산
    side_score = {s: min(conf[f'{s}_{p}'] for p in ('shoulder', 'hip', 'knee', 'ankle'))
                  for s in ('l', 'r')}
    s = max(side_score, key=side_score.get)
    knee = hip = torso = None
    wrist_above_hip = False
    if side_score[s] >= KP_CONF:
        sh, hp, kn, an = (pts[f'{s}_{p}'] for p in ('shoulder', 'hip', 'knee', 'ankle'))
        knee = calculate_angle(hp, kn, an)
        hip = calculate_angle(sh, hp, kn)
        torso = calculate_angle(sh, hp, (hp[0], hp[1] - 10))
        # 이미지 좌표는 y 가 아래로 커지므로 '위' 는 y 가 더 작은 것
        wrist_above_hip = any(ok(w) and pts[w][1] < hp[1] for w in ('l_wrist', 'r_wrist'))

    # 한쪽 발(발목/뒤꿈치/발끝 중 가장 높은 점)이 반대쪽 무릎보다 위인지
    def foot_top(side):
        ys = [pts[n][1] for n in (f'{side}_ankle', f'{side}_heel', f'{side}_foot') if ok(n)]
        return min(ys) if ys else None

    gaps = []
    for foot_side, knee_side in (('l', 'r'), ('r', 'l')):
        fy = foot_top(foot_side)
        if fy is not None and ok(f'{knee_side}_knee') and box_h > 0:
            gaps.append((pts[f'{knee_side}_knee'][1] - fy) / box_h)
    foot_gap = max(gaps) if gaps else None

    return Features(knee, hip, torso, wrist_above_hip, foot_gap)


def classify(f, base, ratio=1.0, foot_margin=FOOT_MARGIN):
    """
    base: (무릎, 엉덩이) 베이스라인 각도.
    1차 판정은 ratio < 1, foot_margin < FOOT_MARGIN 으로 기준을 느슨하게 적용.
    """
    if f.foot_gap is not None and f.foot_gap >= foot_margin:
        return 'Jump'
    if base is None or f.knee is None:
        return 'Normal'
    dk = abs(f.knee - base[0])
    dh = abs(f.hip - base[1])
    if dk >= THETA * ratio and f.wrist_above_hip:
        return 'Jump'
    if dk >= CRAWL_KNEE * ratio and dh >= CRAWL_HIP * ratio and f.torso >= CRAWL_TORSO * ratio:
        return 'Crawl'
    return 'Normal'

"""
YOLO-pose(기본 판정) + MediaPipe(의심 후보 정밀 확인) 기반 Jump/Crawl 탐지

흐름
  1. YOLO11-pose + ByteTrack 으로 모든 사람의 박스/ID/17개 keypoint 를 한 번에 추출
  2. ID 별 베이스라인(평상시 무릎/엉덩이 각도)과 비교해 느슨한 기준으로 1차 판정
  3. 1차에서 걸린 후보만 MediaPipe(33개 keypoint, 발끝/뒤꿈치 포함)로 정밀 확인
  4. CONFIRM_FRAMES 프레임 연속 확인되면 이벤트 확정 → 로그 (쿨다운 동안 중복 기록 X)
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

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
BASELINE_FRAMES = 30       # ID 별 베이스라인 수집 프레임 수
CONFIRM_FRAMES = 3         # 연속 확인 프레임 수
COOLDOWN_FRAMES = 90       # 같은 ID, 같은 이벤트 재기록 금지 프레임 수
STALE_FRAMES = 60          # 이 프레임 수 동안 안 보인 ID 는 상태 삭제
MAX_MP_PER_FRAME = 3       # 프레임당 MediaPipe 호출 상한 (비용 상한)
MIN_MP_BOX_H = 80          # 이보다 작은 박스는 MediaPipe 가 부정확하므로 YOLO 결과로 판정
CROP_PAD = 0.2

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
COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
STATUS_COLOR = {'Normal': (0, 255, 0), 'Jump': (0, 0, 255), 'Crawl': (255, 0, 0)}

logger = logging.getLogger('events')


# ---------------------------------------------------------------- 특징 계산

@dataclass
class Features:
    knee: float | None          # 무릎 각도 (엉덩이-무릎-발목)
    hip: float | None           # 엉덩이 각도 (어깨-엉덩이-무릎)
    torso: float | None         # 상체가 수직에서 기울어진 각도
    wrist_above_hip: bool
    foot_above_knee: bool


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / mag
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def extract_features(pts, conf, box_h):
    """pts/conf: {이름: (x, y)} / {이름: 신뢰도}. YOLO, MediaPipe 공통으로 사용."""
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

    margin = FOOT_MARGIN * box_h
    foot_above_knee = False
    for foot_side, knee_side in (('l', 'r'), ('r', 'l')):
        fy = foot_top(foot_side)
        if fy is not None and ok(f'{knee_side}_knee') and fy < pts[f'{knee_side}_knee'][1] - margin:
            foot_above_knee = True

    return Features(knee, hip, torso, wrist_above_hip, foot_above_knee)


def classify(f, base, ratio=1.0):
    if f.foot_above_knee:
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


# ---------------------------------------------------------------- 사람별 상태

@dataclass
class PersonState:
    sum_knee: float = 0.0
    sum_hip: float = 0.0
    n_base: int = 0
    streak_label: str = 'Normal'
    streak: int = 0
    status: str = 'Normal'
    last_seen: int = 0
    last_event: dict = field(default_factory=dict)

    @property
    def baseline(self):
        if self.n_base < BASELINE_FRAMES:
            return None
        return self.sum_knee / self.n_base, self.sum_hip / self.n_base

    def add_baseline(self, f):
        if self.n_base < BASELINE_FRAMES and f.knee is not None and not f.foot_above_knee:
            self.sum_knee += f.knee
            self.sum_hip += f.hip
            self.n_base += 1

    def update(self, label):
        """연속 프레임 확인. 확정된 이벤트면 이벤트명을, 아니면 None 을 반환."""
        if label == 'Normal':
            self.streak_label, self.streak, self.status = 'Normal', 0, 'Normal'
            return None
        self.streak = self.streak + 1 if label == self.streak_label else 1
        self.streak_label = label
        if self.streak >= CONFIRM_FRAMES:
            self.status = label
            return label
        return None


# ---------------------------------------------------------------- MediaPipe 확인기

class MediaPipeConfirmer:
    """후보가 처음 생길 때만 모델을 로드. 단일 인스턴스를 메인 스레드에서만 사용."""

    def __init__(self, complexity):
        self.complexity = complexity
        self._pose = None
        self.calls = 0

    def landmarks(self, frame, box):
        if self._pose is None:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,       # 후보일 때만 띄엄띄엄 호출하므로 추적 모드 X
                model_complexity=self.complexity,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
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
        return pts, conf

    def close(self):
        if self._pose is not None:
            self._pose.close()


# ---------------------------------------------------------------- 그리기

def blur_background(frame, boxes):
    # 축소 → 블러 → 확대 : 전체 해상도 블러보다 훨씬 가벼움
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))
    out = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (w, h))
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out


def draw_skeleton(img, kxy, kconf):
    for a, b in COCO_EDGES:
        if kconf[a] >= KP_CONF and kconf[b] >= KP_CONF:
            cv2.line(img, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])), (255, 255, 255), 2)
    for (x, y), c in zip(kxy, kconf):
        if c >= KP_CONF:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)


# ---------------------------------------------------------------- 메인

def parse_args():
    p = argparse.ArgumentParser(description='YOLO-pose + MediaPipe Jump/Crawl 탐지')
    p.add_argument('--source', default='sample.mp4', help='영상 경로 또는 웹캠 번호')
    p.add_argument('--model', default='yolo11n-pose.pt', help='YOLO pose 모델 (n/s/m...)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--mp-complexity', type=int, default=1, choices=(0, 1, 2),
                   help='MediaPipe 모델 크기 (0=lite, 가장 가벼움)')
    p.add_argument('--log', default='event_log.txt')
    p.add_argument('--blur', action='store_true', help='사람 외 배경 블러')
    p.add_argument('--skeleton', action='store_true', help='YOLO 스켈레톤 표시')
    p.add_argument('--no-show', action='store_true', help='화면 출력 없이 실행')
    return p.parse_args()


def setup_logger(path):
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def main():
    args = parse_args()
    setup_logger(args.log)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'영상 열기 실패: {args.source}')
        return

    model = YOLO(args.model)
    confirmer = MediaPipeConfirmer(args.mp_complexity)
    states: dict[int, PersonState] = {}
    frame_num = 0
    fps_ema = None
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break
            t0 = time.perf_counter()
            frame_num += 1

            r = model.track(frame, persist=True, classes=[0], conf=CONF_THRES, iou=NMS_THRES,
                            imgsz=args.imgsz, tracker='bytetrack.yaml', verbose=False)[0]

            people = []  # (id, box, kxy, kconf)
            if r.boxes.id is not None and r.keypoints is not None:
                ids = r.boxes.id.int().tolist()
                boxes = r.boxes.xyxy.int().tolist()
                kxy = r.keypoints.xy.cpu().numpy()
                kconf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                         else np.ones(kxy.shape[:2], dtype=np.float32))
                people = list(zip(ids, boxes, kxy, kconf))

            mp_budget = MAX_MP_PER_FRAME
            for pid, box, kp_xy, kp_conf in people:
                st = states.setdefault(pid, PersonState())
                st.last_seen = frame_num
                box_h = box[3] - box[1]

                pts = {n: tuple(kp_xy[i]) for n, i in YOLO_KP.items()}
                conf = {n: float(kp_conf[i]) for n, i in YOLO_KP.items()}
                f_yolo = extract_features(pts, conf, box_h)
                st.add_baseline(f_yolo)
                base = st.baseline

                # 1차: YOLO, 느슨한 기준
                if classify(f_yolo, base, SCREEN_RATIO) == 'Normal':
                    st.update('Normal')
                    continue

                # 2차: 후보만 MediaPipe 로 정밀 확인 (불가하면 YOLO + 원래 기준으로 판정)
                label = None
                if mp_budget > 0 and box_h >= MIN_MP_BOX_H:
                    mp_budget -= 1
                    lm = confirmer.landmarks(frame, box)
                    if lm is not None:
                        label = classify(extract_features(*lm, box_h), base)
                if label is None:
                    label = classify(f_yolo, base)

                event = st.update(label)
                if event and frame_num - st.last_event.get(event, -COOLDOWN_FRAMES) >= COOLDOWN_FRAMES:
                    st.last_event[event] = frame_num
                    knee = f'{f_yolo.knee:.1f}' if f_yolo.knee is not None else '-'
                    hip = f'{f_yolo.hip:.1f}' if f_yolo.hip is not None else '-'
                    logger.info(f'Frame {frame_num} | Person {pid}: {event} - Knee={knee}, Hip={hip}')

            for pid in [p for p, s in states.items() if frame_num - s.last_seen > STALE_FRAMES]:
                del states[pid]

            dt = time.perf_counter() - t0
            fps_ema = 1 / dt if fps_ema is None else fps_ema * 0.9 + (1 / dt) * 0.1

            if not args.no_show:
                vis = blur_background(frame, [p[1] for p in people]) if args.blur else frame.copy()
                for pid, (x1, y1, x2, y2), kp_xy, kp_conf in people:
                    status = states[pid].status
                    color = STATUS_COLOR[status]
                    thick = 1 if status == 'Normal' else 3
                    if args.skeleton:
                        draw_skeleton(vis, kp_xy, kp_conf)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
                    cv2.putText(vis, f'ID {pid} | {status}', (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis, f'FPS {fps_ema:.1f}  MP calls {confirmer.calls}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('YOLO-pose + MediaPipe Jump/Crawl', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        confirmer.close()

    elapsed = time.perf_counter() - t_start
    if frame_num:
        print(f'완료: {frame_num} 프레임, 평균 {frame_num / elapsed:.1f} FPS, '
              f'MediaPipe 호출 {confirmer.calls}회 ({confirmer.calls / frame_num:.2f}/프레임). '
              f'로그는 {args.log}에 저장.')


if __name__ == '__main__':
    main()

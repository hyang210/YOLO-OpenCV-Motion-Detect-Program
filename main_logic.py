"""
YOLO-pose(기본 판정) + MediaPipe(의심 후보 정밀 확인) 기반 Jump/Crawl 탐지

흐름
  1. YOLO11-pose + ByteTrack 으로 모든 사람의 박스/ID/17개 keypoint 를 한 번에 추출
  2. ID 별 베이스라인(평상시 무릎/엉덩이 각도)과 비교해 느슨한 기준으로 1차 판정
  3. 각도 변화는 YOLO 값으로 원래 기준 재판정, 발이 무릎 근처까지 올라온 후보만
     MediaPipe(발끝/뒤꿈치 포함 33개 keypoint)로 발-무릎 높이를 정밀 확인
  4. 일정 시간 연속 확인되면 이벤트 확정 → 로그 (쿨다운 동안 중복 기록 X)

설정값은 detector/config.py 에 있음
"""
import argparse
import logging
import time
from dataclasses import replace

import cv2
import numpy as np
from ultralytics import YOLO

from detector.config import (CONF_THRES, FOOT_SCREEN_MARGIN, MAX_MP_PER_FRAME, MIN_MP_BOX_H,
                             NMS_THRES, SCREEN_RATIO, FrameParams)
from detector.confirmer import MediaPipeConfirmer
from detector.features import YOLO_KP, classify, extract_features
from detector.person_state import PersonState
from detector.visualize import blur_background, draw_hud, draw_person, draw_skeleton

logger = logging.getLogger('events')


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


def read_people(r):
    """YOLO track 결과 → [(id, box, keypoint xy, keypoint conf)]"""
    if r.boxes.id is None or r.keypoints is None:
        return []
    ids = r.boxes.id.int().tolist()
    boxes = r.boxes.xyxy.int().tolist()
    kxy = r.keypoints.xy.cpu().numpy()
    if r.keypoints.conf is not None:
        kconf = r.keypoints.conf.cpu().numpy()
    else:
        kconf = np.ones(kxy.shape[:2], dtype=np.float32)
    kconf[(kxy == 0).all(axis=-1)] = 0  # 검출 안 된 점은 (0, 0) 으로 나옴
    return list(zip(ids, boxes, kxy, kconf))


def main():
    args = parse_args()
    setup_logger(args.log)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'영상 열기 실패: {args.source}')
        return

    params = FrameParams.from_fps(cap.get(cv2.CAP_PROP_FPS))
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
            people = read_people(r)

            # 1차: YOLO keypoint 로 느슨하게 판정해서 후보 선별
            candidates = []
            for pid, box, kp_xy, kp_conf in people:
                st = states.get(pid)
                if st is None:
                    st = states[pid] = PersonState(params)
                st.last_seen = frame_num

                pts = {n: tuple(kp_xy[i]) for n, i in YOLO_KP.items()}
                conf = {n: float(kp_conf[i]) for n, i in YOLO_KP.items()}
                f_yolo = extract_features(pts, conf, box[3] - box[1])
                st.add_baseline(f_yolo)

                if classify(f_yolo, st.baseline, SCREEN_RATIO, FOOT_SCREEN_MARGIN) == 'Normal':
                    st.update('Normal')
                else:
                    candidates.append((pid, box, f_yolo, st))

            # 2차: 원래 기준으로 최종 판정.
            #  - 각도 변화: 베이스라인과 같은 모델인 YOLO 값으로만 비교 (모델 간 관절 위치 차이 배제)
            #  - 발-무릎 높이: 발이 무릎 근처면 MediaPipe 발끝/뒤꿈치 값으로 교체
            #    (이미 연속 판정 중인 후보부터, 예산 초과·실패·다른 사람 검출 시 YOLO 값 유지)
            candidates.sort(key=lambda c: c[3].streak, reverse=True)
            mp_budget = MAX_MP_PER_FRAME
            for pid, box, f_yolo, st in candidates:
                box_h = box[3] - box[1]
                f_final = f_yolo
                near_knee = f_yolo.foot_gap is not None and f_yolo.foot_gap >= FOOT_SCREEN_MARGIN
                if near_knee and mp_budget > 0 and box_h >= MIN_MP_BOX_H:
                    mp_budget -= 1
                    lm = confirmer.landmarks(frame, box)
                    if lm is not None:
                        f_mp = extract_features(*lm, box_h)
                        if f_mp.foot_gap is not None:
                            f_final = replace(f_yolo, foot_gap=f_mp.foot_gap)
                label = classify(f_final, st.baseline)

                event = st.update(label)
                if event and st.should_log(event, frame_num):
                    knee = f'{f_yolo.knee:.1f}' if f_yolo.knee is not None else '-'
                    hip = f'{f_yolo.hip:.1f}' if f_yolo.hip is not None else '-'
                    logger.info(f'Frame {frame_num} | Person {pid}: {event} - Knee={knee}, Hip={hip}')

            for pid in [p for p, s in states.items() if frame_num - s.last_seen > params.stale]:
                del states[pid]

            dt = time.perf_counter() - t0
            fps_ema = 1 / dt if fps_ema is None else fps_ema * 0.9 + (1 / dt) * 0.1

            if not args.no_show:
                vis = blur_background(frame, [p[1] for p in people]) if args.blur else frame.copy()
                for pid, box, kp_xy, kp_conf in people:
                    if args.skeleton:
                        draw_skeleton(vis, kp_xy, kp_conf)
                    draw_person(vis, pid, box, states[pid].status)
                draw_hud(vis, fps_ema, confirmer.calls)
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

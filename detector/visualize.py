"""화면 출력용 그리기 함수"""
import cv2

from .config import KP_CONF

COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
STATUS_COLOR = {'Normal': (0, 255, 0), 'Jump': (0, 0, 255), 'Crawl': (255, 0, 0)}


def blur_background(frame, boxes):
    # 축소 → 블러 → 확대 : 전체 해상도 블러보다 훨씬 가벼움
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)))
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


def draw_person(img, pid, box, status):
    x1, y1, x2, y2 = box
    color = STATUS_COLOR[status]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if status == 'Normal' else 3)
    cv2.putText(img, f'ID {pid} | {status}', (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_hud(img, fps, mp_calls):
    cv2.putText(img, f'Proc FPS {fps:.1f}  MP calls {mp_calls}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

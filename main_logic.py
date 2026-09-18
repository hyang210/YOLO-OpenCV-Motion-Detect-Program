import cv2
import math
import datetime
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# YOLO 모델 및 임계값 설정
MODEL_PATH = 'C:/pythonPractice/yolo11m.pt'
CONF_THRES = 0.5
NMS_THRES = 0.4

# 점프 및 기기 동작 탐지를 위한 기준값 설정
THETA = 80.0
FOOT_OFF = 15
CRAWL_KNEE_HIP = 30.0
CRAWL_PELVIS_DROP = 10
CRAWL_TORSO_TH = 60.0

# 추적자 및 ID 초기화
trackers = {}
next_id = 0
model = YOLO(MODEL_PATH)


# 사람 외 영역에 흐림 효과 적용 함수
def apply_blur_to_background(frame, boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    blurred_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        blurred_color[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return blurred_color


# IOU 계산 함수
def IOU(box1, box2):
    x1, y1, x2, y2 = box1
    x_1, y_1, x_2, y_2 = box2
    xi1, yi1 = max(x1, x_1), max(y1, y_1)
    xi2, yi2 = min(x2, x_2), min(y2, y_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (x_2 - x_1) * (y_2 - y_1) - inter_area
    return inter_area / union_area if union_area > 0 else 0


# NMS (Non-Maximum Suppression) 함수
def non_max_suppression_persons(boxes, threshold=0.4):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    selected = []
    for box in boxes:
        if all(IOU(box, s) < threshold for s in selected):
            selected.append(box)
    return selected


# MediaPipe를 이용한 포즈 각도 계산 클래스
class PoseAngleEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            enable_segmentation=False
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        magnitude_ba = math.hypot(*ba)
        magnitude_bc = math.hypot(*bc)

        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0

        angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        return math.degrees(angle_rad)

    def extract_landmark(self, landmarks, idx, image_shape):
        h, w = image_shape
        try:
            landmark = landmarks[idx]
            return (int(landmark.x * w), int(landmark.y * h))
        except IndexError:
            return (0, 0)

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            hip = self.extract_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value, (h, w))
            knee = self.extract_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value, (h, w))
            ankle = self.extract_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value, (h, w))
            shoulder = self.extract_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, (h, w))
            vertical = (hip[0], hip[1] - 10)

            knee_angle = self.calculate_angle(hip, knee, ankle)
            hip_angle = self.calculate_angle(shoulder, hip, knee)
            vertical_angle = self.calculate_angle(vertical, hip, knee)

            print(f"knee angle : {knee_angle}")
            print(f"hip angle : {hip_angle}")
            print(f"vertical_angle : {vertical_angle}")

            return frame, results


# 한 명의 사람에 대해 동작을 분석하고 결과 리턴
def process_person(crop_img, box, frame_num, person_id, estimator, baseline, frame, offset):
    offset_x, offset_y = offset

    with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:

        results = pose.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None

    # 스켈레톤 그리기
    if results.pose_landmarks:
        estimator.mp_drawing.draw_landmarks(
            crop_img,
            results.pose_landmarks,
            estimator.mp_pose.POSE_CONNECTIONS
        )

        # 전체 프레임 기준으로 keypoint 좌표 보정 후 프레임에 그림
        h_crop, w_crop, _ = crop_img.shape
        lm = results.pose_landmarks.landmark
        # visibility < 0.5 인 경우에는 unreliable 한 좌표이므로 skip
        required = [estimator.mp_pose.PoseLandmark.LEFT_HIP,
                    estimator.mp_pose.PoseLandmark.LEFT_KNEE,
                    estimator.mp_pose.PoseLandmark.LEFT_ANKLE]

        if any(lm[p].visibility < 0.5 for p in required):  # 추가된 내용. 정확도가 떨어지면 mediapipe 표시하지 않기
            return None

        keypoints = []
        for pt in lm:
            x = int(pt.x * w_crop) + offset_x
            y = int(pt.y * h_crop) + offset_y
            keypoints.append((x, y))
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        for start_idx, end_idx in estimator.mp_pose.POSE_CONNECTIONS:
            if keypoints[start_idx] != (0, 0) and keypoints[end_idx] != (0, 0):
                cv2.line(frame, keypoints[start_idx], keypoints[end_idx], (255, 255, 255), 2)

        lm = results.pose_landmarks.landmark
        h, w, _ = crop_img.shape
        px = lambda pt: (pt.x * w, pt.y * h)

        hip = px(lm[estimator.mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = px(lm[estimator.mp_pose.PoseLandmark.LEFT_KNEE])
        left_ankle = px(lm[estimator.mp_pose.PoseLandmark.LEFT_ANKLE])
        left_wrist = px(lm[estimator.mp_pose.PoseLandmark.LEFT_WRIST])
        right_ankle = px(lm[estimator.mp_pose.PoseLandmark.RIGHT_ANKLE])
        right_knee = px(lm[estimator.mp_pose.PoseLandmark.RIGHT_KNEE])

        shoulder = px(lm[estimator.mp_pose.PoseLandmark.LEFT_SHOULDER])
        vertical = (hip[0], hip[1] - 10)

        knee_ang = estimator.calculate_angle(hip, left_knee, left_ankle)
        hip_ang = estimator.calculate_angle(shoulder, hip, left_knee)
        torso_ang = estimator.calculate_angle(shoulder, hip, vertical)

        # 베이스라인 수집
        if frame_num <= 30:
            baseline['sum_knee'] += knee_ang
            baseline['sum_hip'] += hip_ang
            baseline['sum_ankle'] += left_ankle[1]
            baseline['count'] += 1
            return None

        if baseline['count'] >= 30 and not baseline.get('initialized'):
            baseline['knee'] = baseline['sum_knee'] / baseline['count']
            baseline['hip'] = baseline['sum_hip'] / baseline['count']
            baseline['ankle'] = baseline['sum_ankle'] / baseline['count']
            baseline['initialized'] = True

        if not baseline.get('initialized'):
            return {
                'box': box,
                'person_id': person_id,
                'status': 'Normal',
                'crop_img': crop_img
            }

        # 동작 판단
        delta_k = abs(knee_ang - baseline['knee'])
        delta_h = abs(hip_ang - baseline['hip'])
        delta_a = abs(left_ankle[1] - baseline['ankle'])
        hand_above_hip = (left_wrist[1] > hip[1])
        foot_above_knee = (right_ankle[1] < left_knee[1]) or (left_ankle[1] < right_knee[1])

        if (delta_k >= THETA and hand_above_hip) or foot_above_knee:
            log_event("Jump", frame_num, person_id, knee_ang, hip_ang, delta_k)
            status = "Jump"
        elif delta_k >= CRAWL_KNEE_HIP and delta_h >= CRAWL_PELVIS_DROP and torso_ang >= CRAWL_TORSO_TH:
            log_event("Crawl", frame_num, person_id, knee_ang, hip_ang, delta_k)
            status = "Crawl"
        else:
            status = "Normal"

        if status != "Normal":
            log_event(status, frame_num, person_id, knee_ang, hip_ang, delta_k)

        keypoints = {
            'LEFT_HIP': hip,
            'LEFT_KNEE': left_knee,
            'LEFT_ANKLE': left_ankle,
            'LEFT_SHOULDER': shoulder
        }

        return {
            'box': box,
            'person_id': person_id,
            'status': status,
            'crop_img': crop_img
        }


# 로그 기록 함수
def log_event(event_type, frame_num, person_id, knee_ang, hip_ang, score):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("event_log.txt", "a") as f:
        f.write(
            f"[{timestamp}] Frame {frame_num} | Person {person_id}: {event_type} - Knee={knee_ang:.1f}, Hip={hip_ang:.1f}, Score={score:.1f}\n")


# 메인 함수
def main():
    global next_id, trackers
    video_path = 'C:/pythonPractice/sample.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("영상 파일 열기 실패.")
        return

    estimator = PoseAngleEstimator()
    baseline = {'sum_knee': 0, 'sum_hip': 0, 'sum_ankle': 0, 'count': 0}
    frame_num = 0
    executor = ThreadPoolExecutor(max_workers=4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("프레임 읽기 실패")
            break

        frame_num += 1
        keypoint_futures = []

        results = model.predict(frame, conf=CONF_THRES, iou=NMS_THRES, verbose=False)[0]
        boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id].lower() == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

        filtered_boxes = non_max_suppression_persons(boxes)
        frame = apply_blur_to_background(frame, filtered_boxes)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, "person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        updated_trackers = {}
        used_ids = set()

        new_trackers = {}
        matched_id = set()

        for box in filtered_boxes:
            best_iou = 0
            matched = None
            for pid, prev_box in trackers.items():
                iou = IOU(box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    matched = pid
            if best_iou > 0.4 and matched not in matched_id:
                new_trackers[matched] = box
                matched_id.add(matched)
            else:
                new_trackers[next_id] = box
                matched_id.add(next_id)
                next_id += 1

        trackers = new_trackers

        futures = []

        for person_id, (x1, y1, x2, y2) in trackers.items():
            pad = 0.4
            dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
            xA, yA = max(0, x1 - dx), max(0, y1 - dy)
            xB, yB = min(frame.shape[1], x2 + dx), min(frame.shape[0], y2 + dy)

            if xB <= xA or yB <= yA:
                continue

            area = (xB - xA) * (yB - yA)
            if area < 6400:
                continue

            crop = frame[yA:yB, xA:xB].copy()

            if (yB - yA) / frame.shape[0] < 0.1 or (xB - xA) / frame.shape[1] < 0.03:
                continue

            future = executor.submit(
                process_person,
                crop,
                (x1, y1, x2, y2),
                frame_num,
                person_id,
                estimator,
                baseline,
                frame,
                (xA, yA)
            )
            futures.append(future)

        for future in futures:
            # 오류 원인 탐지 코드 ----
            try:
                result = future.result()
            except Exception as e:
                # 이 줄이 없으면 여기서 터진 예외는 잡히지 않고 프로그램 전체가 꺼집니다.
                print(f"[Error] process_person 에러: {e}")
                continue
            # -----여기까지------
            if result is not None:
                x1, y1, x2, y2 = result['box']
                label = f"ID {result['person_id']} | {result['status']}"

                if result is not None:
                    x1, y1, x2, y2 = result['box']
                    person_id = result['person_id']
                    status = result['status']
                    label = f"ID {person_id} | {status}"

                    if status == 'Jump':
                        color, thickness = (0, 0, 255), 3
                    elif status == 'Crawl':
                        color, thickness = (255, 0, 0), 3
                    else:
                        continue

                if result['status'] == 'Normal' and 'crop_img' in result:
                    pad = 0.4
                    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
                    xA, yA = max(0, x1 - dx), max(0, y1 - dy)
                    xB, yB = min(frame.shape[1], x2 + dx), min(frame.shape[0], y2 + dy)
                if (result['crop_img'].shape[0] == (yB - yA)
                    and result['crop_img'].shape[1] == (xB - xA)):
                    frame[yA:yB, xA:xB] = result['crop_img']

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLO + MediaPipe + Jump/Crawl", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
    print("완료. 로그는 event_log.txt에 저장.")


if __name__ == '__main__':
    main()
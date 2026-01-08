import cv2
import numpy as np
from ultralytics import YOLO
import cvzone


# Mouse callback function for RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load YOLO11 model
model = YOLO("yolo11n.pt")
names = model.names

# Open the video file or webcam
cap = cv2.VideoCapture('peoplecount1.mp4')

# --- VIDEO EXPORT INITIALIZATION START ---
frame_width = 1020
frame_height = 600
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30  # Fallback if FPS cannot be read

# Define the codec and create VideoWriter object
# 'mp4v' for .mp4 or 'XVID' for .avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_people_count.mp4', fourcc, fps, (frame_width, frame_height))
# --- VIDEO EXPORT INITIALIZATION END ---

count = 0
area1 = [(250, 444), (211, 448), (473, 575), (514, 566)]
area2 = [(201, 449), (177, 453), (420, 581), (457, 577)]
enter = {}
exits = {}
list1 = []
list2 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            c = names[class_id]
            if 'person' in c:
                x1, y1, x2, y2 = box

                # for entries
                result0 = cv2.pointPolygonTest(np.array(area2, np.int32), (x1, y2), False)
                if result0 >= 0:
                    enter[track_id] = (x1, y2)

                if track_id in enter:
                    result1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x1, y2), False)
                    if result1 >= 0:
                        if list1.count(track_id) == 0:
                            list1.append(track_id)

                # for exits
                result2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x1, y2), False)
                if result2 >= 0:
                    exits[track_id] = (x1, y2)

                if track_id in exits:
                    result3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x1, y2), False)
                    if result3 >= 0:
                        if list2.count(track_id) == 0:
                            list2.append(track_id)

                # Draw boxes for all tracked people
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)

    enter_in_shop = len(list1)
    exit_from_shop = len(list2)
    cvzone.putTextRect(frame, f'Entered: {enter_in_shop}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Exited: {exit_from_shop}', (50, 100), 2, 2)
    cvzone.putTextRect(frame, f'Current: {enter_in_shop - exit_from_shop}', (50, 140), 2, 2)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)

    # --- WRITE FRAME TO FILE ---
    out.write(frame)
    # ---------------------------

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()  # CRITICAL: Release the writer to save the file correctly
cv2.destroyAllWindows()
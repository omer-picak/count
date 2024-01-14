import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture("peoplecount1.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

area2 = [(280, 259), (276, 363), (389, 354), (389, 256)]
line = (379, 0, 379, 497)
going_out = {}
counter1 = []
entering = set()
paths = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)

    # Create a blank canvas
    canvas = np.zeros_like(frame)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        # Store path history
        if id not in paths:
            paths[id] = []
        paths[id].append((cx, cy))

        # Draw path on the canvas
        if len(paths[id]) > 1:
            cv2.polylines(canvas, [np.array(paths[id], np.int32)], False, (0, 255, 0), 2)

        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        result = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
        print(result)
        if result >= 0:
            going_out[id] = (cx, cy)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        
        if id in going_out and result<0:
            entering.add(id)
                       
    # Draw the line on the canvas
    

    # Check for intersection with the line
    

    # Draw the canvas on the frame
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    print(entering)
    print(len(entering))

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

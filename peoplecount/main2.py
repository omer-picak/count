import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*



model=YOLO('yolov8s.pt')




def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture("peoplecount1.mp4")


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
tracker=Tracker()
#area1=[(432,384),(289,390),(474,469),(609,458)]
#area2=[(279,392),(168,415),(233,497),(454,469)]

area1=[(312,388),(289,390),(474,469),(497,462)]
area2=[(279,392),(250,397),(423,477),(454,469)]

"""going_out={}
counter1=[]
going_in={}
counter2=[]"""
tracker = Tracker()
entering = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    takip_edilen_liste = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        if 'person' in c:
            takip_edilen_liste.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(takip_edilen_liste)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)

        result = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)

        if result >= 0:
            cv2.circle(frame, (x4, y4), 4, (255, 255, 0), -1)
            entering.add(id)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in entering:
            result1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if result1 >= 0:
                cv2.circle(frame, (x4, y4), 4, (0, 255, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 255), 2)
    print(entering)
    print(len(entering))

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
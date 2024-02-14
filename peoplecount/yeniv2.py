import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import time

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

def check_inside_rect(bbox, rect_area):
    x, y = bbox[0], bbox[1]
    if rect_area[0][0] < x < rect_area[1][0] and rect_area[0][1] < y < rect_area[1][1]:
        return True
    return False

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vidp.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
persondown={}
tracker=Tracker()
counter1=[]
rect_area = [(383, 170), (645, 275)]  # Dikdörtgen alanın koordinatları

inside_people = {}

cy1=194
cy2=220
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    list=[]

    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        
        if check_inside_rect((cx, cy), rect_area):
            if id not in inside_people:
                inside_people[id] = time.time()  # İçeri giriş zamanı
            inside_time = time.time() - inside_people[id]  # İçerde kaldığı süre
            cvzone.putTextRect(frame, f'ID: {id} | Sure: {inside_time:.2f} sn', (x3, y3), 1, 2)
        else:
            if id in inside_people:
                # Dikdörtgen alanı terk ettiğinde hesaplamaları yap
                inside_time = time.time() - inside_people[id]  # İçerde kaldığı süre
                print(f"ID {id} içerde {inside_time:.2f} saniye kaldı.")
                inside_people.pop(id)  # Kişiyi listeden çıkar
                
        if cy1<(cy+offset) and cy1>(cy-offset):
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
            persondown[id]=(cx,cy)

        if id in persondown:
            if cy2<(cy+offset) and cy2>(cy-offset):
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2) 
                if counter1.count(id)==0:
                    counter1.append(id)

    cv2.line(frame,(3,cy1),(1018,cy1),(0,255,0),2)
    cv2.line(frame,(5,cy2),(1019,cy2),(0,255,255),2)
    cv2.rectangle(frame, (383, 170), (645, 275), color=(255,0,0), thickness=2)

    cvzone.putTextRect(frame,f'Person In={len(counter1)}',(50,60),2,2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

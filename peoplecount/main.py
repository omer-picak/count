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
#area1=[(463,220),(428,244),(576,273),(600,222)]
#area2=[(401,262),(374,287),(546,345),(564,297)]

area1=[(312,388),(289,390),(474,469),(497,462)]

area2=[(279,392),(250,397),(423,477),(454,469)]

going_out={}
counter1=[]
going_in={}
counter2=[]
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
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        result=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if result>=0:
            
            going_out[id]=(cx,cy)
        if id in going_out:
            result1=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if result1>=0:
                cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2) 
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                if counter1.count(id)==0:
                    counter1.append(id)
        result2=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result2>=0:
            going_in[id]=(cx,cy)
        if id in going_in:
            result3=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if result3>=0:
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2) 
                    cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                    if counter2.count(id)==0:
                        counter2.append(id)
 
    cv2.polylines(frame, [np.array(area1,np.int32)],True,(255,0,255),2)
    cv2.polylines(frame, [np.array(area2,np.int32)],True,(255,0,255),2)
    
    g_out=len(counter1)
    g_in=len(counter2)
    cvzone.putTextRect(frame,f'PERSON_OUT:{g_out}',(50,60),2,2)
    cvzone.putTextRect(frame,f'PERSON_IN:{g_in}',(50,140),2,2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

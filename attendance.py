import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Attendance'
image = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    cuImg = cv2.imread(f'{path}/{cl}')
    image.append(cuImg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def encodingimage(image):
    Encodelist = []
    for img in image:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        Encodelist.append(encoding)
    return Encodelist

def markattendance(name):
    with open('Attendance.csv','r+') as f:
        mydatalist= f.readlines()
        namelist= []
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now= datetime.now()
            datestring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')



encodedlist = encodingimage(image)
print('Encoded')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCuFrame = face_recognition.face_locations(imgS)
    encodeCuFrame = face_recognition.face_encodings(imgS,faceCuFrame)

    for enocdeFac ,Facloc in zip(encodeCuFrame,faceCuFrame):
        mathes=face_recognition.compare_faces(encodedlist,enocdeFac)
        facdis=face_recognition.face_distance(encodedlist,enocdeFac)
        print(facdis)
        matchIndex = np.argmin(facdis)

        if mathes[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = Facloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img,(x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markattendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
import cv2
import numpy as np
import face_recognition

imgVk=face_recognition.load_image_file('ImageTest/Virat kohli.jpg')
imgVk=cv2.cvtColor(imgVk,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('ImageTest/vktest.jfif')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

face_loc=face_recognition.face_locations(imgVk)[0]
encodeVK=face_recognition.face_encodings(imgVk)[0]
cv2.rectangle(imgVk,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,0,300),2)

face_loc=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,0,200),2)

result=face_recognition.compare_faces([encodeVK],encodetest)
facedis=face_recognition.face_distance([encodeVK],encodetest)
cv2.putText(imgtest,f'{result},{round(facedis[0],2)}',(50,50),cv2.FONT_ITALIC,1,(255,0,255),2)
print(result,facedis)


cv2.imshow('Virat',imgVk)
cv2.imshow('VK test',imgtest)
cv2.waitKey(0)


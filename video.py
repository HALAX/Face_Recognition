import cv2
import numpy as np 

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)


while True:
    check, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detect.detectMultiScale(gray, 1.25, 5)

    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        frame = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

        cv2.imshow("Doraemon Eye", img)
        if(cv2.waitKey(1) ==ord('q')):
            break

            cam.release()
            cv2.destroyAllWindows()
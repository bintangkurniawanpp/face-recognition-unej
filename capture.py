import cv2
import sys
from pathlib import Path

cpt = 0  
maxFrames = 1000
video = cv2.VideoCapture(0)
v = video.get(5)
fps = video.get(cv2.CAP_PROP_FPS)
inp = input("Input Nama : ")
while True:
    ret,frame = video.read()
    cv2.imshow('camera',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        dir = Path("dataset/" + str(inp)).mkdir(parents=True, exist_ok=True)
        while cpt < maxFrames:
            if not ret:
                sys.exit(0)
            cv2.imshow('captured', frame)
            cv2.imwrite("dataset/" + str(inp) + "/image%04i.jpg" %cpt, frame)
            
            cpt += 1

video.release()
cv2.destroyAllWindows()
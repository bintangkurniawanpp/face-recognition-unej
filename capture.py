import cv2
import sys

cpt = 0
maxFrames = 100 # if you want 5 frames only.

video = cv2.VideoCapture(1)

while True:
    ret,frame = video.read()
    cv2.imshow('camera',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        while cpt < maxFrames:
            if not ret: # if return code is bad, abort.
                sys.exit(0)
            cv2.imshow('captured', frame)
            cv2.imwrite("dataset/muchdor/image%04i.jpg" %cpt, frame)
            cpt += 1
    
video.release()
cv2.destroyAllWindows()
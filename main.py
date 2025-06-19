import cv2
import numpy as np
import imutils 
import time 

#export QT_QPA_PLATFORM=xcb


def load_and_preprocess(img_path):
    
    image = cv2.resize(img_path, (1280,720))
    gray_image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    return image, gray_image


def subtract_images(image1,image2):
    diff = cv2.absdiff(image1,image2)
    _,thresh = cv2.threshold(diff, 85, 255, cv2.THRESH_BINARY)

    return diff, thresh



cam = cv2.VideoCapture(0)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', 500, 300)

static_frame = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

i = 0
out = cv2.VideoWriter(f'output{i}.mp4', fourcc, 20.0, (1280, 720))

f = 0

frame_count = 0

while True:


   
    if(i >= 50):
        break

    ret, frame = cam.read()
    
    image, gray_image = load_and_preprocess(frame)

    if static_frame is None:
        static_frame = gray_image
        continue

    

    diff,thresh = subtract_images(static_frame,gray_image)
    dialeted_image = cv2.dilate(thresh,None,iterations=2)

    cnts = cv2.findContours(dialeted_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    

    for c in cnts:
        if cv2.contourArea(c) < 700:
            continue
        
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(image,(x,y), (x+w,y+h), (0,255,0), 2)
    

    if (not (len(cnts) == 0)):
        out.write(image)
        frame_count+=1
        f = 1
    
    if(f == 1):
        if (len(cnts) == 0) and (frame_count >= 10):
            f = 0
            frame_count = 0
            out.release()
            i+=1
            out = cv2.VideoWriter(f'output{i}.mp4', fourcc, 20.0, (1280, 720))
            print("MOTION OVER")


       
    
    cv2.imshow('Camera', image)
    
    
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.1)

out.release()
cam.release()
cv2.destroyAllWindows()
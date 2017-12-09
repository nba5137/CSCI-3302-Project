## For chasing Sparki robot(blue color) by using webcam. 
## Modified original code in OpemCV tutorial: 
## http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html

import numpy as np
import cv2
import serial

cap = cv2.VideoCapture(0)
## Added a variable for cell index.
cell_index = 0;

## Added function to output cell index in a txt file.
def printout(index, filename):
  f = open(filename,'w')
  f.seek(0) ##Overwrite for updating index
  sindex = str(index)
  f.write(sindex)
  f.close()

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

## Changed: Reset HSV boundaries here, in order to chase blue objects.
mask = cv2.inRange(hsv_roi, np.array((0.,0.,25.)), np.array((0.,0.,255.)))

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        

        ## Added: setting coordinate.
        ## Setting left-top corner as (0,0) by subtracting
        ## Thus the center of left-top corner frame is (62.5,45)
        ## Screen size: 515 * 390.
        dX = x
        dY = y
        ## Finding cell index. 
        if (dX < 172 and dY < 130):
            cell_index = 0
        if (172 < dX and dX < 343 and dY < 130):
            cell_index = 1
        if (343 < dX and dY < 130):
            cell_index = 2
        if (dX < 172 and 130 < dY and dY < 260):
            cell_index = 3
        if (172 < dX and dX < 343 and 130 < dY and dY < 260):
            cell_index = 4
        if (343 < dX and 130 < dY and dY< 260):
            cell_index = 5
        if (dX < 172 and 260 < dY):
            cell_index = 6
        if (172 < dX and dX < 343 and 260 < dY):
            cell_index = 7
        if (343 < dX and 260 < dY):
            cell_index = 8

        
        ## Added: print info on screen and store index in a txt file.
        ## original position for putText: (10, frame.shape[0] - 10)
        cv2.putText(frame, "X: {}, Y: {}, Cell: {}".format(dX, dY, cell_index),
		(20, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1, (0, 0, 255), 2)

        ## Storing...
        printout(cell_index, "output.txt")
        
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()

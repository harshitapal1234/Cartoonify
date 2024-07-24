# Importing all necessary libraries
import cv2
import os
import numpy as np

# Read the video from specified path
cam = cv2.VideoCapture("/home/habilelabs/Desktop/video/WhatsApp Video 2022-04-16 at 10.49.13 AM.mp4")

try:
	
	# creating a folder named data
	if not os.path.exists('data'):
		os.makedirs('data')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')

# frame
currentframe = 0

while(True):
        ret,cartoon = cam.read()
        if ret:
            name = './data/frame' + str(currentframe) + '.jpg'

            print ('Creating...' + name)
            cv2.imwrite(name, cartoon)
            currentframe += 1
        else:
            break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
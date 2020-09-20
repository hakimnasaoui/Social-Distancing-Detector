#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from itertools import combinations
import math
import time
import argparse


# In[2]:


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(WORKING_DIR, "input") 
OUTPUT_FOLDER = os.path.join(WORKING_DIR, "output") 

# Video source here : https://www.youtube.com/watch?v=GJNjaRJWVP8  
# Cutted by clideo https://clideo.com/ : cliped from 00:16.50s to 00:25.00s

# There are 3 different video quality/resolutions : 
# test-video-480p / test-video-720p / test-video-1080p
INPUT_FILENAME = os.path.join(INPUT_FOLDER, "test-video-480p.mp4") 

# Save processed frames into folder 'frames'
SAVE_FRAMES = False
FRAMES_DIR = os.path.join(OUTPUT_FOLDER, "frames")

# Save processed frames as a new video
SAVE_VIDEO = False
OUT_VIDEO_FILENAME = os.path.join(OUTPUT_FOLDER, "output-video-hog.avi")

# Colors in BGR
POSITIF_DISTANCING_DETECTION_COLOR = (0, 255, 0) # Green 
NEGATIF_DISTANCING_DETECTION_COLOR = (0, 0, 255) # Red (in BGR)
DETECTION_LINE_WIDTH = 2

# Defining unsafe distance threshold
DISTANCE_THRESHOLD = 100.0


# In[ ]:


def parseArguments():
    # Creating argument parser obj
    parser = argparse.ArgumentParser(description='Social Distancing Detector using HOG')

    # Adding arguments
    parser.add_argument("-i", "--input", help="Input video filename.", 
                        type=str, default=INPUT_FILENAME)
    parser.add_argument("-t", "--threshold", help="Distance threshold.", 
                        type=float, default=DISTANCE_THRESHOLD)

    parser.add_argument("-sf", "--saveFrames", help="Save processed frames.", 
                        type=bool, default=SAVE_FRAMES)
    parser.add_argument("-sv", "--saveVideo", help="Save processed video.", 
                        type=bool, default=SAVE_VIDEO)
    
    # Parse arguments
    args = parser.parse_args()

    return args


# In[ ]:


args = parseArguments()
# print("You are running the script under the following arguments : ")    
# for a in args.__dict__:
#     print(str(a) + ": " + str(args.__dict__[a]))
        
INPUT_FILENAME = args.input

if args.threshold <= 0 :
    print("Argument 'threshold' value must be greater than 0.")
else:
    DISTANCE_THRESHOLD = args.threshold
    
SAVE_FRAMES = args.saveFrames
SAVE_VIDEO = args.saveVideo

# if args.saveVideo.lower() not in ["true", "false", 1, 0]:
#     print("Argument 'saveFrames' value is not set properly please enter a bool value or 1/0.")
# elif args.saveFrames.lower() in ["true", 1]:
#     SAVE_FRAMES = True
# else:
#     SAVE_FRAMES = False
    
# if args.saveVideo.lower() not in ["true", "false", 1, 0]:
#     print("Argument 'saveVideo' value is not set properly please enter a bool value or 1/0.")
# elif args.saveVideo.lower() in ["true", 1]:
#     SAVE_VIDEO = True
# else:
#     SAVE_VIDEO = False

print("You are running this script under the following setup:")
print ("INPUT_FILENAME = ", INPUT_FILENAME)
print ("DISTANCE_THRESHOLD = ", DISTANCE_THRESHOLD)
print ("SAVE_FRAMES = ", SAVE_FRAMES)
print ("SAVE_VIDEO = ", SAVE_VIDEO)


# In[3]:


if SAVE_FRAMES == True:
    os.makedirs(FRAMES_DIR, exist_ok=True)


# In[4]:


# Initialize the HOG descriptor detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# In[5]:


start_time = time.time()

cap = cv2.VideoCapture(INPUT_FILENAME)
frames_counter=0

# initalize video output func (VideoWriter object) 
if SAVE_VIDEO == True:
    # converting video resolution from float to int.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create VideoWriter for the output video
    out = cv2.VideoWriter(OUT_VIDEO_FILENAME, 
                          cv2.VideoWriter_fourcc('M','J','P','G'), 
                          10, 
                          (frame_width,frame_height))

# chech if the video exists and able to open 
if cap.isOpened() == False:
    print('Error opening filename')
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_counter+=1
        
        # Saving frames as 1 channel in order to speed up the process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people in the frame
        (detected_humans, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                                                padding=(2, 2))

        boxes = dict()
        i=1 # for iterating between boxes
        
        # Looping through all RoI of the frame
        for (x,y,w,h) in detected_humans:
            
            # Saving frames boxes infos center and coordination of each RoI
            x_center = int(x + w/2) # int((2*x+w)/2.0)
            y_center = int(y + h/2)
            boxes[i] = {"center":(x_center, y_center), "coord":(x, y, x+w, y+h)}
            i+=1

            people_unsafe_distance = []
            
            # looping though all RoIs found in this frame by taking couple of RoIs each time    
            for (p1, box1), (p2, box2) in combinations(boxes.items(), 2):
                
                # Calc the distance between two persons p1 and p2 
                (center_p1, coord_p1) = box1["center"], box1["coord"]
                (center_p2, coord_p2) = box2["center"], box2["coord"]
                
                distance = math.sqrt((center_p1[0] - center_p2[0]) ** 2 
                                     + (center_p1[1] - center_p2[1]) ** 2)
                
                # Check if this distance is lower than the Threshold
                if distance < DISTANCE_THRESHOLD:
                    if p1 not in people_unsafe_distance:
                        people_unsafe_distance.append(p1)
                    if p2 not in people_unsafe_distance:
                        people_unsafe_distance.append(p2)
            
            # Draw the bounding box of RoI according to the threshold  
            for id, box in boxes.items():
                (center, coord) = box["center"], box["coord"]
                if id in people_unsafe_distance:
                    cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), 
                                  NEGATIF_DISTANCING_DETECTION_COLOR, DETECTION_LINE_WIDTH)
                else:
                    cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), 
                                  POSITIF_DISTANCING_DETECTION_COLOR, DETECTION_LINE_WIDTH)
        
        # Posibility to save the processed frames into a folder  
        if SAVE_FRAMES == True:
            cv2.imwrite(FRAMES_DIR + '/hog-frame-' + str(frames_counter) + ".jpg", frame)

        # Display the result as a video (looping frame by frame)
        cv2.imshow("Detecting ... (press ENTER to stop)", frame)
        
        # Posibility to write the frame into video OUT_VIDEO_FILENAME
        if SAVE_VIDEO == True:
            out.write(frame)
    
        # Posibility to stop processing the video frames using a key
        if cv2.waitKey(1) == 13: # 13 is ENTER Key
            break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
print("--- Process completed in : %s seconds ---" % round(end_time - start_time))


# In[6]:


# print(detected_humans) # detected humans (RoI) on the last processed frame"
# print(boxes) # bounding boxes of the last processed frame"


import os
import json
import sys
import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
import torch
from pathlib import Path
import pandas as pd
from util_art import detector_utils as detector_utils
from util_art import alertcheck2 as alertcheck2
from util_art import alertcheck as alertcheck
from util_art.detector_utils import distance_to_camera
from datetime import date

import numpy as np


lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())


if __name__ == '__main__':
# Load YOLOv5 model
    repository_owner = "ultralytics"  # Replace with the correct repository owner
    repository_name = "yolov5"  # Replace with the correct repository name
    branch = "master"  # Replace with the correct branch name (or omit this argument if using the default branch)

    model = torch.hub.load(r"C:\Users\vishn\Downloads\new_pr\yolov5", "custom", path="C:\\Users\\vishn\\Downloads\\new_pr\\yolov5\\best.pt", source="local", force_reload=True)
    
    model.eval()
    # Process video frames
    # Set the source as 0 for webcam or provide the path to your video file

    a=b=0

    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    cap = cv2.VideoCapture(0)   # Replace 'input_video.mp4' with your video file

    hand_cnt=0

    score_thresh = 0.50

    #Oriendtation of machine    
    Orientation= 'bt'
	#input("Enter the orientation of hand progression ~ lr,rl,bt,tb :")

    #For Machine
    #Line_Perc1=float(input("Enter the percent of screen the line of machine :"))
    Line_Perc1=float(15)

    #For Safety
    #Line_Perc2=float(input("Enter the percent of screen for the line of safety :"))
    Line_Perc2=float(30)

    # max number of hands we want to detect/track
    

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    while True:
    # Read a frame from the webcam
        ret, frame = cap.read()
        frame = np.array(frame)

        if not ret:
            break

        if im_height == None:
            im_height, im_width = frame.shape[:2]

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error Converting to RGB")
        
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        

        # Perform object detection using YOLOv5
        results = model(frame)

        # Post-process the detected hands
        pred_boxes = results.pred[0][:, :4]  # Bounding boxes (x1, y1, x2, y2)
        pred_scores = results.pred[0][:, 4]   # Objectness scores
        pred_classes = results.pred[0][:, 5]  # Predicted classes

        num_hands_detect = len(pred_scores)

        Line_Position1,Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_Perc2)  


        # a,b=detector_utils.draw_box_on_image(
        #         num_hands_detect, score_thresh, pred_scores, pred_boxes, pred_classes, im_width, im_height, frame,Line_Position2,Orientation)
        # lst1.append(a)
        # lst2.append(b)

        # no_of_time_hand_detected=no_of_time_hand_crossed=0
        # # Calculate Frames per second (FPS)
        # num_frames += 1
        # elapsed_time = (datetime.datetime.now() -
        #                 start_time).total_seconds()
        # fps = num_frames / elapsed_time

        # Process detected objects (e.g., draw boxes on the frame)
        for box, score, class_id in zip(pred_boxes, pred_scores, pred_classes):
            box = box.int().cpu().numpy()  # Convert to numpy array
            class_id = int(class_id)
            class_name = model.names[class_id]  # Get class name from model
            confidence = float(score)
            x1, y1, x2, y2 = box  # Coordinates of the bounding box

            # Draw bounding box on the frame
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display class name and confidence score
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            p1 = (int(x1),int(y1))
            p2 = (int(x2),int(y2))

            dist = distance_to_camera(avg_width, focalLength, int(x2-x1))

            if dist:
                hand_cnt=hand_cnt+1   

            a=alertcheck.drawboxtosafeline(frame,p1,p2,Line_Position2,Orientation)
            k=alertcheck2.drawboxtosafeline(frame,p1,p2,Line_Position1,Orientation)

            if hand_cnt==0 :
                b=0
                #print(" no hand")
            else:
                b=1

            lst1.append(a)
            lst2.append(b)

        # Display the frame with detected objects
        cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            cap.stop()
            break
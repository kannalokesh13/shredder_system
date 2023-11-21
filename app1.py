import os
import json
import sys
print(os.getcwd())
os.chdir('C:\\Users\\vishn\\Downloads\\new_pr\\yolov5')
print(os.getcwd())

import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from pathlib import Path
import pandas as pd
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
    model = attempt_load(weights='best.pt')
    model.eval()

    # Process video frames
    source = 0  # Set the source as 0 for webcam or provide the path to your video file


    cap = cv2.VideoCapture(source)  # Replace 'input_video.mp4' with your video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Perform object detection on the frame
        results = model(frame)

        # Get bounding boxes, scores, and classes
        pred_boxes = results.pred[0][:, :4]  # Bounding boxes (x1, y1, x2, y2)
        pred_scores = results.pred[0][:, 4]   # Objectness scores
        pred_classes = results.pred[0][:, 5]  # Predicted classes

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

        # Display the frame with detected objects
        cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            cap.stop()
            break

    # Release the video capture object and close all OpenCV windows
    

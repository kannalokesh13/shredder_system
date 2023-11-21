import os
import json
import subprocess
import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
import torch
# from yolov5.models.experimental import attempt_load

import pandas as pd
from datetime import date

import numpy as np

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())


# def save_data(no_of_time_hand_detected, no_of_time_hand_crossed):

#     try:   
#         today = date.today()
#         today=str(today)

      
#         rb = xlrd.open_workbook('result.xls')
#         sheet = rb.sheet_by_index(0) 
#         sheet.cell_value(0, 0) 
      
         
#         #print(sheet.nrows)
#         q=sheet.cell_value(sheet.nrows-1,1)
        
#         rb = xlrd.open_workbook('result.xls') 
#         #rb = xlrd.open_workbook(loc) 
#         wb=copy(rb)
#         w_sheet=wb.get_sheet(0)
        
#         if q==today:
#             w=sheet.cell_value(sheet.nrows-1,2)
#             e=sheet.cell_value(sheet.nrows-1,3)
#             w_sheet.write(sheet.nrows-1,2,w+no_of_time_hand_detected)
#             w_sheet.write(sheet.nrows-1,3,e+no_of_time_hand_crossed)
#             wb.save('result.xls')      
#         else:
#             w_sheet.write(sheet.nrows,0,sheet.nrows)
#             w_sheet.write(sheet.nrows,1,today)
#             w_sheet.write(sheet.nrows,2,no_of_time_hand_detected)
#             w_sheet.write(sheet.nrows,3,no_of_time_hand_crossed)
#             wb.save('result.xls')
#     except FileNotFoundError:
#         today = date.today()
#         today=str(today)
         

#         # Workbook is created 
#         wb = Workbook() 

#         # add_sheet is used to create sheet. 
#         sheet = wb.add_sheet('Sheet 1') 

#         sheet.write(0, 0, 'Sl.No')
#         sheet.write(0, 1, 'Date') 
#         sheet.write(0, 2, 'Number of times hand detected') 
#         sheet.write(0, 3, 'Number of times hand crossed') 
#         m=1
#         sheet.write(1, 0, m)
#         sheet.write(1, 1, today) 
#         sheet.write(1, 2, no_of_time_hand_detected) 
#         sheet.write(1, 3, no_of_time_hand_crossed) 

#         wb.save('result.xls')
        
if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    
    #vs = cv2.VideoCapture('rtsp://192.168.1.64')
    
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
    num_hands_detect = 2

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    
    def count_no_of_times(lst):
        x=y=cnt=0
        for i in lst:
            x=y
            y=i
            if x==0 and y==1: 
                cnt=cnt+1
        return cnt 
    try:
        
        


        vs = VideoStream(0).start()
        
        while True:
            frame = vs.read()

            cv2.imwrite('yolov5/frame_images/frame.jpg', frame)
            
            yolo_command = "cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../yolov5/frame_images/frame.jpg"
            subprocess.run(yolo_command, shell=True)

            output = os.popen(yolo_command).read()

            # Parse the JSON output to extract bounding boxes, scores, and classes
            # detections = json.loads(output)
            lines = output.strip().split("\n")
            detections = []
            for line in lines[1:]:
                class_name, confidence, *bbox = line.split()
                bbox = list(map(float, bbox))
                detections.append({
                    "label": class_name,
                    "confidence": float(confidence),
                    "box": {
                        "xmin": int(bbox[0]),
                        "ymin": int(bbox[1]),
                        "xmax": int(bbox[2]),
                        "ymax": int(bbox[3])
                    }
                })

            # Save detections to a JSON file
            with open('detections.json', 'w') as json_file:
                json.dump(detections, json_file)

            # frame = np.array(frame)

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            #cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            
            #cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            
            # Run image through tensorflow graph

            # boxes, scores, classes = detector_utils.detect_objects(
            #     frame, detection_graph, sess)
            
            Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_Perc2) 


            # Process the detections and draw bounding boxes on the frame
            for detection in detections:
                class_name = detection['label']
                confidence = detection['confidence']
                bbox = detection['box']

                # Extract coordinates of the bounding box


                x, y, w, h = bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']

                # Draw the bounding box on the frame
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Display class name and confidence score
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the frame with detected objects


            # Draw bounding boxeses and text
            # a,b=detector_utils.draw_box_on_image(
            #     num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame,Line_Position2,Orientation)
            # lst1.append(a)
            # lst2.append(b)
            # no_of_time_hand_detected=no_of_time_hand_crossed=0
            # # Calculate Frames per second (FPS)
            # num_frames += 1
            # elapsed_time = (datetime.datetime.now() -
            #                 start_time).total_seconds()
            # fps = num_frames / elapsed_time

            # if args['display']:
            
                # Display FPS on frame
                # detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows() 
                vs.stop()
                break
        
        # no_of_time_hand_detected=count_no_of_times(lst2)
        # #no_of_time_hand_detected=b
        # no_of_time_hand_crossed=count_no_of_times(lst1)
        # #print(no_of_time_hand_detected)
        # #print(no_of_time_hand_crossed)
        # save_data(no_of_time_hand_detected, no_of_time_hand_crossed) 
        # print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt: 
        # no_of_time_hand_detected=count_no_of_times(lst2)
        # no_of_time_hand_crossed=count_no_of_times(lst1)
        # today = date.today()
        # save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        # print("Average FPS: ", str("{0:.2f}".format(fps)))
        print('got some error')


    
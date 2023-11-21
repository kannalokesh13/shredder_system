import cv2
from playsound import playsound
import pygame
import pandas as pd

#crossed=0

def drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation):
    
    #global crossed 
    if(Orientation=="bt"):
     bounding_mid=(int((p1[0]+p2[0])/2),int(p1[1]))   
     if(bounding_mid):
         cv2.line(img=image_np, pt1=bounding_mid, pt2=(bounding_mid[0],Line_Position2), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
         distance_from_line=bounding_mid[1]-Line_Position2
    elif(Orientation=="tb"):
     bounding_mid=(int((p1[0]+p2[0])/2),int(p2[1]))   
     if(bounding_mid):
         cv2.line(img=image_np, pt1=bounding_mid, pt2=(bounding_mid[0],Line_Position2), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
         distance_from_line=Line_Position2-bounding_mid[1]
    elif(Orientation=="lr"):
     bounding_mid=(int(p2[0]),int((p1[1]+p2[1])/2))   
     if(bounding_mid):
         cv2.line(img=image_np, pt1=bounding_mid, pt2=(Line_Position2,bounding_mid[1]), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
         distance_from_line=Line_Position2-bounding_mid[0]     
    elif(Orientation=="rl"):
     bounding_mid=(int(p1[0]),int((p1[1]+p2[1])/2))
     if(bounding_mid):
         cv2.line(img=image_np, pt1=bounding_mid, pt2=(Line_Position2,bounding_mid[1]), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
         distance_from_line=bounding_mid[1]-Line_Position2
    
    if (distance_from_line <= 0) :
            
             #crossed+=1
             pygame.init()
             posii=int(image_np.shape[1]/2)        
             cv2.putText(image_np, "ALERT", (posii, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0,0), 2)
			 #sound = os.path.join()
             # Load the sound file
             pygame.mixer.music.load("util_art/alert.wav")

             # Play the sound file
             pygame.mixer.music.play()

             # Wait for the sound to finish playing (optional)
             while pygame.mixer.music.get_busy():
                 pygame.time.Clock().tick(10)

             # Quit pygame (when you're done with audio playback)
             pygame.quit()


            #  playsound("util_art/alert.wav")
             cv2.rectangle(image_np, (posii-20,20), (posii+85,60), (255,0,0), thickness=3, lineType=8, shift=0)
             #to write into xl-sheet            
             return 1
    else:
        return 0
    
   



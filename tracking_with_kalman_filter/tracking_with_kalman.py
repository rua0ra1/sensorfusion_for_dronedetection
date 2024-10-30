## this example is referred from https://medium.com/@siromermer/predicting-objects-motion-with-kalman-filter-and-fast-algorithm-2278c551670b
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time 
import os



#path to video
vid_path=os.path.join(".", "data","helicopter.mp4")
video = cv2.VideoCapture(vid_path)

if not video.isOpened():
    print("Error: Could not open video file.")

#read only the first frame for drawing a rectanle for desired object
ret,frame = video.read()

# # I am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
# x_min,y_min,x_max,y_max=36000,36000,0,0

# def coordinat_chooser(event,x,y,flags,param):
#     global go , x_min , y_min, x_max , y_max

#     # when you click the right button, it will provide coordinates for variables
#     if event==cv2.EVENT_LBUTTONDOWN:
        
#         # if current coordinate of x lower than the x_min it will be new x_min , same rules apply for y_min 
#         x_min=min(x,x_min) 
#         y_min=min(y,y_min)

#          # if current coordinate of x higher than the x_max it will be new x_max , same rules apply for y_max
#         x_max=max(x,x_max)
#         y_max=max(y,y_max)

#         # draw rectangle
#         cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),1)


#     """
#         if you didn't like your rectangle (maybe if you made some misscliks),  reset the coordinates with the middle button of your mouse
#         if you press the middle button of your mouse coordinates will reset and you can give a new 2-point pair for your rectangle
#     """
#     if event==cv2.EVENT_MBUTTONDOWN:
#         print("reset coordinate  data")
#         x_min,y_min,x_max,y_max=36000,36000,0,0

# cv2.namedWindow('coordinate_screen')
# # Set mouse handler for the specified window, in this case, "coordinate_screen" window
# cv2.setMouseCallback('coordinate_screen',coordinat_chooser)

# while True:
#     cv2.imshow("coordinate_screen",frame) # show only first frame 
    
#     k = cv2.waitKey(5) & 0xFF # after drawing rectangle press ESC   
#     if k == 27:
#         cv2.destroyAllWindows()
#         break

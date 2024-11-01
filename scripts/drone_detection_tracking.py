from ultralytics import YOLO
import cv2
import os
import numpy as np



# Load the YOLOv8 model

model_path = os.path.join("..", "yolo_weights/best.pt") 
model = YOLO(model_path)

# Detection function
def getDronePositionFromImg(img):

    # Run inference
    results = model(img)

    # Process and display results
    for result in results:
        boxes = result.boxes  # Each box contains box information, confidence, and class
        
        xc=0
        yc=0
        drone_detected=False
        for box in boxes:
            # Box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{model.names[class_id]} ({confidence:.2f})"


            if(model.names[class_id]=="drone"):
                xc=round((x1+x2)/2)
                yc=round((y1+y2)/2)
                drone_detected=True

    return xc,yc,drone_detected






# Startcapturing the video from file
vid_path=os.path.join("..","Drone-detection-dataset/Data/Video_V/V_DRONE_022.mp4")
cap = cv2.VideoCapture(vid_path)

# Get the original video frame rate (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")


# initialize the kalman filter
kalman = cv2.KalmanFilter(4, 2)   
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 0.033 , 0], [0, 1, 0, 0.033 ], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5  # Measurement noise


#write results to a video

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
out = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict the new position of the ball
    predicted = kalman.predict()
    predicted_x, predicted_y = int(predicted[0]), int(predicted[1])
    predicted_dx, predicted_dy = predicted[2], predicted[3]  # Predicted velocity

    print(predicted_x, predicted_y )
    print(f"Predicted velocity: (dx: {predicted_dx}, dy: {predicted_dy})")

    
    # Detect the ball in the current frame
    xc,yc,drone_detected = getDronePositionFromImg(frame)
    
    if drone_detected:
        measured_x=xc 
        measured_y = yc
        # Correct the Kalman Filter with the actual measurement
        kalman.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
        # Draw the detected ball
        cv2.circle(frame, (measured_x, measured_y), 6, (0, 255, 0), 2) # green --> correct position
    
    # Draw the predicted position (Kalman Filter result)
    cv2.circle(frame, (predicted_x, predicted_y), 8, (0, 0, 255), 2) # red --> predicted position

    # Show the frame
    cv2.imshow("drone Tracking", frame)

    # Write the frame to the output video
    out.write(frame)
    
    # Break on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 30 ms delay for smooth playback
        break

cap.release()
out.release()  # Save and close the output video
cv2.destroyAllWindows()

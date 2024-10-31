import cv2
import os


# Function to extract frames from a video and save them
def extract_frames(video_path, output_dir, basename, capture_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * capture_rate)  # Interval based on capture rate and FPS
    frame_count = 0
    saved_count = 0

    # Loop to read and save each frame at the specified interval
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at the specified interval
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f'{basename}_{saved_count}.jpg')
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} with capture rate {capture_rate} seconds.")
    return saved_count





def generateImages():
    # function loop through all the videos and generate img from it
    data_dir = os.path.join("..", "Drone-detection-dataset/Data/Video_IR") 
    video_dir = data_dir

    output_image_dir = os.path.join("..", "training_data/Video_IR", "images")
    output_label_dir = os.path.join("..", "training_data/Video_IR", "labels")
    class_mapping = {"AIRPLANE": 0, "BIRD": 1, "DRONE": 2, "HELICOPTER": 3}
     # Process each video and its corresponding .mat file
    for video_file in os.listdir(video_dir):
      if video_file.endswith('.mp4'):
        basename = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)

        # Extract frames from video
        frame_count = extract_frames(video_path, output_image_dir, basename)



if __name__ == "__main__":
        generateImages()
        print("completed the generating the images from video") 
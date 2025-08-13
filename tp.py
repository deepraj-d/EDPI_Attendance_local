from src.process_frames import get_data
import cv2
import os
from multiprocessing import Process
import time

# ---- Video paths ----
VIDEO_PATHS = {
    "ENTRANCE": "D:/fast/file1.mp4",
    "EXIT": "D:/fast/output.mp4"  # Example second video path
}

def process_video_stream(cam_name, video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video stream for {cam_name}")
        return

 
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        
        get_data(frame, cam_name) 
        
        
    cap.release()
    

if __name__ == "__main__":
    processes = []
    
    for cam_name, video_path in VIDEO_PATHS.items():
        p = Process(target=process_video_stream, args=(cam_name, video_path))
        processes.append(p)
        p.start()
        time.sleep(0.1)  

    for p in processes:
        p.join()

    print("All video processing completed")
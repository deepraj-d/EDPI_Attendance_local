from src.process_frames import get_data
import cv2
import os
from utils import time_stamp_region #, get_timestamp

# ---- Setup video input/output ----
path_video = "D:/cctv/full_day.mp4" 
file_name = os.path.splitext(os.path.basename(path_video))[0]
CAM = "OUT"

def calculate_chunksize(total_frames, num_processes):
    """
    Calculate an optimal chunksize for multiprocessing frame processing.
    Formula: (total_frames / num_processes) / 100
    Ensures at least 1 chunk size.
    """
    chunksize = max(1, int((total_frames / num_processes) / 100))
    return chunksize


cap = cv2.VideoCapture(path_video)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (frame_width, frame_height)

writer = cv2.VideoWriter(
   f"output_data/{file_name}.mp4",
   cv2.VideoWriter_fourcc(*'mp4v'),
   fps,
   size
)

# ---- Seeker update function ----
def on_trackbar(val):
   cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# ---- Create seeker bar ----
cv2.namedWindow("Office_Entrance_Cam")
cv2.createTrackbar("Seek", "Office_Entrance_Cam", 0, total_frames - 1, on_trackbar)

frame_count = 0
door_open = False

while True:
   ret, frame = cap.read()

   if not ret:
       break

   frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
   get_data(frame, cam=CAM)
   cv2.imwrite('avc.jpg',frame)
   cv2.imshow("Office_Entrance_Cam", frame)
   writer.write(frame)

   # update trackbar position
   cv2.setTrackbarPos("Seek", "Office_Entrance_Cam", frame_count)

   key = cv2.waitKey(1) & 0xFF
   if key == ord('q'):
       break
   elif key == ord('f'):  # fast-forward 100 frames
       new_pos = min(total_frames - 1, frame_count + 100)
       cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
   elif key == ord('r'):  # rewind 100 frames
       new_pos = max(0, frame_count - 100)
       cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

cap.release()
writer.release()
cv2.destroyAllWindows()



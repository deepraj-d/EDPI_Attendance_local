from multiprocessing import Pool, cpu_count
import cv2
import os
from src.process_frames import get_data

# ---- Setup video ----
path_video = "D:/cctv/full_day.mp4"
file_name = os.path.splitext(os.path.basename(path_video))[0]
CAM = "OUT"
def process_frame(args):
    frame, frame_id = args
    get_data(frame, cam=CAM)
    return None

def frame_generator(path_video):
    cap = cv2.VideoCapture(path_video)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield (frame, frame_id)
        frame_id += 1
    cap.release()

def main():
    physical_cores = cpu_count() // 2
    num_processes = max(1, physical_cores - 1)  # leave 1 core free
    print(f"Using {num_processes} worker processes")

    chunksize = 100  # safe memory usage
    print(f"Processing in chunks of {chunksize} frames per worker")

    with Pool(processes=num_processes) as pool:
        for _ in pool.imap(process_frame, frame_generator(path_video), chunksize=chunksize):
            pass

if __name__ == "__main__":
    main()

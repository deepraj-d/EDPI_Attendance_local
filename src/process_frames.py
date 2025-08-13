from ultralytics import YOLO
import os
import cv2
import csv
import torch
from utils import get_employee_name,get_time,get_employee_name_arcface ,get_timestamp
from src.embeddings import db_path,load_db
from src.embeddings import pad_crop


# --- GPU Configuration ---
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")  # Use first GPU
torch.backends.cudnn.benchmark = True  # Optimize CUDA


yolo_model_face = YOLO('pre_trained_models/yolov11n-face.pt')
yolo_model_body = YOLO('pre_trained_models/yolov8n.pt')

from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# CSV file path
csv_file_path = "sample.csv"


# Ensure CSV header is written only once
if not os.path.exists(csv_file_path):
   with open(csv_file_path, mode='w', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(['Name','TimeStamp',"In_out"])  # header


# ---- Embeddings ----
known_faces = load_db(db_path)


def get_data(frame,cam):
   new_entries = []
   results = yolo_model_body(frame,verbose = False)
   for result in results:
       boxes = result.boxes
       for i, box in enumerate(boxes):
           cls_id = int(box.cls.item())
           if cls_id != 0:  # 0 is 'person' in COCO
               continue

           conf = box.conf.item()
           if conf < 0.8:
               continue

           x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
           person_crop = frame[y1:y2, x1:x2]

           # face_found = False
           x1_f = y1_f = x2_f = y2_f = 0

           # ------- face detection --------
           if person_crop.size > 0:
              
              
               results2 = yolo_model_face(person_crop,verbose = False)
               for res in results2:
                   boxes_face = res.boxes
                  
                   for i,b in enumerate(boxes_face):
                      
                       conf_face = b.conf.item()
                       if conf_face < 0.8:
                           continue

                       # Face coordinates relative to the person_crop
                       fx1, fy1, fx2, fy2 = map(int, b.xyxy[0].tolist())

                       # Convert to original frame coordinates
                       x1_f, y1_f = x1 + fx1, y1 + fy1
                       x2_f, y2_f = x1 + fx2, y1 + fy2
                       cv2.rectangle(frame,(x1_f,y1_f),(x2_f,y2_f),color=(0,255,0),thickness=2)
                       cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                       padded_face = pad_crop(frame, x1_f, y1_f, x2_f, y2_f, padding=20)

                       name,score = get_employee_name_arcface(padded_face,known_faces=known_faces,model_app=app)
                      
                       # if name is not None and score is not None:
                       if name is not None and score is not None:
                           name = name.split("_")[0] if "_" in name else name
                           new_entries.append([name,
                                               get_timestamp(fr=frame),
                                               cam])
                          
                       cv2.putText(frame, name, (x2_f, y2_f + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                           # save_img_to_cluster(name,person_crop,name,frame_count)
                           # print(f"\033[91mPerson Identified {name} saved to cluster\033[0m")
   # Save new data to CSV
   if new_entries:
       with open(csv_file_path, mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerows(new_entries)

  
   return None





          




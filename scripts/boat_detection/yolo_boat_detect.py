from ultralytics import YOLO
import cv2
import json
import os
import time

# %%
# Модель (COCO pre-trained, містить клас "boat")
model = YOLO("yolov8s.pt")

# Шлях до папки з фото
input_dir = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL"
output_dir = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL_output_yolo"
os.makedirs(output_dir, exist_ok=True)

# %%
results_json = []

# %%
def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    return inter_w * inter_h

def box_area(box):
    return max(0, (box[2] - box[0])) * max(0, (box[3] - box[1]))

def filter_overlapping_boxes(boxes, overlap_threshold=0.8):
    keep = []
    for i, boxA in enumerate(boxes):
        areaA = box_area(boxA)
        is_contained = False
        for j, boxB in enumerate(boxes):
            if i == j:
                continue
            areaB = box_area(boxB)
            inter = intersection_area(boxA, boxB)
            overlap_ratio = inter / areaA if areaA > 0 else 0
            if overlap_ratio > overlap_threshold and areaB > areaA:
                is_contained = True
                break
        if not is_contained:
            keep.append(boxA)
    return keep

# %%
image_counter = 0
total_start = time.time()

boat_names = os.listdir(input_dir)
for boat_name in boat_names:
    boat_folder = os.path.join(input_dir, boat_name)
    if not os.path.isdir(boat_folder):
        continue

    for filename in os.listdir(boat_folder):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_counter += 1
        start_time = time.time()

        path = os.path.join(input_dir, boat_folder, filename)
        results = model(path, classes=[8], verbose=False)  # клас 8 = boat у COCO

        boxes = results[0].boxes.xyxy.cpu().numpy()
        boxes = filter_overlapping_boxes(boxes, overlap_threshold=0.8)

        # залишаємо тільки найбільший прямокутник
        if len(boxes) > 0:
            boxes = [max(boxes, key=lambda b: box_area(b))]

        image = cv2.imread(path)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Зберігаємо картинку
        boat_folder_output = os.path.join(output_dir, boat_name)
        os.makedirs(boat_folder_output, exist_ok=True)
        boat_folder_output_file = os.path.join(boat_folder_output, filename)
        cv2.imwrite(boat_folder_output_file, image)

        elapsed_time = time.time() - start_time

        # JSON
        boxes_data = [{"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)} for (x1, y1, x2, y2) in boxes]
        results_json.append({
            "image": filename,
            "file_path": boat_folder_output_file,
            "detections": boxes_data,
            "boat_name": boat_name,
            "boat_count": len(boxes),
            "processing_time_sec": round(elapsed_time, 3)
        })

        print(f"✅ Processed {image_counter}: {boat_name} | {filename} | Time: {elapsed_time:.2f}s")

# Зберігаємо JSON
json_path = os.path.join(output_dir, "boats_detected.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)

total_elapsed = time.time() - total_start
print(f"\n✅ All Done! Processed {image_counter} images in {total_elapsed:.2f}s")
print("\n✅ JSON saved:", json_path)

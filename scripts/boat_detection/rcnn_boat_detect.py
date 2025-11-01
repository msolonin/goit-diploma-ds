import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import os
import json
import time

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# %%
def iou_percentage(inner, outer):
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])

    return inter_area / inner_area if inner_area > 0 else 0

# %%
results_json = []
total_start_time = time.time()

input_dir = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL"
output_dir = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL_output_rcnn"
os.makedirs(output_dir, exist_ok=True)

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat"
]

image_counter = 0

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

        img_path = os.path.join(boat_folder, filename)
        rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        tensor = F.to_tensor(rgb_img).to(device)
        with torch.no_grad():
            pred = model([tensor])[0]

        boxes = pred["boxes"].cpu().numpy().astype(int)
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        filtered_boxes = [
            box for box, score, label in zip(boxes, scores, labels)
            if score > 0.6 and label < len(COCO_CLASSES) and COCO_CLASSES[label] == "boat"
        ]

        clean_boxes = []
        for i, box in enumerate(filtered_boxes):
            keep = True
            for j, other in enumerate(filtered_boxes):
                if i != j and iou_percentage(box, other) > 0.8:
                    if ((other[2]-other[0])*(other[3]-other[1])) >= ((box[2]-box[0])*(box[3]-box[1])):
                        keep = False
                        break
            if keep:
                clean_boxes.append(box)

        if clean_boxes:
            clean_boxes = [max(clean_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))]
        else:
            clean_boxes = []

        out_folder = os.path.join(output_dir, boat_name)
        os.makedirs(out_folder, exist_ok=True)
        img_output_file = os.path.join(out_folder, filename)

        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        for (x1, y1, x2, y2) in clean_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(img_output_file, img)

        elapsed_time = time.time() - start_time

        results_json.append({
            "image": filename,
            "file_path": img_output_file,
            "detections": [
                {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                for (x1, y1, x2, y2) in clean_boxes
            ],
            "boat_name": boat_name,
            "boat_count": len(clean_boxes),
            "processing_time_sec": round(elapsed_time, 3)
        })

        print(f"✅ Processed {image_counter}: {boat_name} | {filename} | Time: {elapsed_time:.2f}s")


json_path = os.path.join(output_dir, "boat_boxes.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)

total_time = time.time() - total_start_time
print(f"\n All Done! Processed: {image_counter} images")
print(f"\n Total time: {total_time:.2f}s | Avg per image: {total_time/image_counter:.2f}s")
print("\n JSON saved:", json_path)

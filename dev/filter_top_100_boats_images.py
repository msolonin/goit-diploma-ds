import os
import shutil


# %%
# === CONFIG ===

TOP_RATING = 100

ORIGINAL_DATASET = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_pb"
OUTPUT_DATASET = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_pb_output"

os.makedirs(OUTPUT_DATASET, exist_ok=True)
COPY = True
# %%


existing_folders = [f for f in os.listdir(ORIGINAL_DATASET) if os.path.isdir(os.path.join(ORIGINAL_DATASET, f))]


folder_image_counts = {}
for boat_folder in existing_folders:
    boat_folder_path = os.path.join(ORIGINAL_DATASET, boat_folder)
    image_files = [f for f in os.listdir(boat_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    folder_image_counts[boat_folder] = len(image_files)

# Get top 100 folders with most images
top_folders = sorted(folder_image_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_RATING]

print("Top 100 folders by number of images:")
for folder, count in top_folders:
    print(folder, count)
            



# %%
if COPY:
    for folder_name, count in top_folders:
        src_path = os.path.join(ORIGINAL_DATASET, folder_name)
        dst_path = os.path.join(OUTPUT_DATASET, folder_name)
        shutil.copytree(src_path, dst_path)
    
    print(f"✅ Copied {len(top_folders)} folders to {OUTPUT_DATASET}")

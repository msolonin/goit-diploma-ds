import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd



image_path =  "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL_output/Hallberg-Rassy 50/b568c7febb97_out.jpg" 



class BoatModelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

model_type = "seal"  # or "seal"
best_model_path = f"best_boat_model_{model_type}.pth"
csv_path = "boat_dataset_3_class1.csv"


df = pd.read_csv(csv_path)
df = df[(df["boat_type"] == model_type) & (df["photo_type"].isin(["out", "boat"]))]

classes = sorted(df["boat_model"].unique().tolist())
print(f"Loaded {len(classes)} boat model classes for type '{model_type}'")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BoatModelClassifier(num_classes=len(classes))
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
img_t = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_t)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

# Get top prediction
pred_idx = probs.argmax()
pred_class = classes[pred_idx]

# Build dictionary of probabilities
result = {cls: round(p * 100, 2) for cls, p in zip(classes, probs)}

print(f"\n Image: {image_path}")
print(f"Predicted Boat Model: {pred_class}")
print(f"Class probabilities (top 5):")

# Show top 5
for cls, prob in sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   {cls:<25} {prob:>6.2f}%")



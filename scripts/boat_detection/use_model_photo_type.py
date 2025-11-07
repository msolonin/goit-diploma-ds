import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn



image_path = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL/Alubat Ovni 450/8c2c6071f9ca.jpg"



class PhotoTypeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# =========================================================
# Load model
# =========================================================
model_path = "best_photo_type_classifier.pth"
class_names = ["boat", "in", "out"]  # must match training order!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhotoTypeClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# =========================================================
# Image preprocessing
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

# =========================================================
# Predict
# =========================================================
with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0]
# =========================================================
# Output
# =========================================================
pred_idx = torch.argmax(probs).item()
pred_class = class_names[pred_idx]

print(f"Predicted class: {pred_class}\n")
print("Class probabilities:")
for cls, p in zip(class_names, probs):
    print(f"  {cls:>5}: {p.item()*100:.2f}%")

max_idx = torch.argmax(probs)
max_class = class_names[max_idx]
max_value = probs[max_idx].item()

print(f"Class with max probability: {max_class} ({max_value:.2f}%)")
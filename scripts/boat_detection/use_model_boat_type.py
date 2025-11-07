import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn



image_path = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL/Alubat Ovni 450/8c2c6071f9ca.jpg"


class BoatTypeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


model_path = "best_boat_type_clasifier.pth"
csv_path = "boat_dataset_3_class1.csv"


classes = ['motor', 'seal']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BoatTypeClassifier(num_classes=len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
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

pred_idx = probs.argmax()
pred_class = classes[pred_idx]

result = {cls: round(p * 100, 2) for cls, p in zip(classes, probs)}

print(f"Image: {image_path}")
print(f"Predicted class: {pred_class}")
print(f"Class probabilities: {result}")


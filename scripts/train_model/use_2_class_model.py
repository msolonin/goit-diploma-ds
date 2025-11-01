# %%
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision

import pandas as pd


# %%


IMAGE_PATH = "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL/A-Yachts a27/68080938da2c.jpg"
# %%


class BoatClassifier(nn.Module):
    def __init__(self, num_boat_models, num_photo_types=2):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # видаляємо оригінальний fc
        self.model_head = nn.Linear(2048, num_boat_models)
        self.photo_head = nn.Linear(2048, num_photo_types)

    def forward(self, x):
        features = self.backbone(x)
        model_logits = self.model_head(features)
        photo_logits = self.photo_head(features)
        return model_logits, photo_logits

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = torch.load("boat_classifier_full.pth")
model_loaded.eval()
model_loaded.to(device)
dataset = "boat_dataset.csv"
df = pd.read_csv(dataset)
boat_model_classes = df["boat_model"].unique().tolist()
photo_type_classes = ["interior", "exterior"]


# %%

# Transform image (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(device)  # add batch dimension

with torch.no_grad():
    model_logits, type_logits = model_loaded(img)
    boat_model_pred = torch.argmax(model_logits, dim=1).item()
    photo_type_pred = torch.argmax(type_logits, dim=1).item()

print("Predicted boat model:", boat_model_classes[boat_model_pred])
print("Predicted photo type:", photo_type_classes[photo_type_pred])
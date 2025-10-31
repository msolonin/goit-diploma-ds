import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision
import torch.optim as optim
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


# %%
dataset = "boat_dataset.csv"
df = pd.read_csv(dataset)

# %%

class BoatDataset(Dataset):
    def __init__(self, csv_file, boat_model_classes, photo_type_classes, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.boat_model_classes = boat_model_classes
        self.photo_type_classes = photo_type_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        model_label = self.boat_model_classes.index(row["boat_model"])
        type_label = self.photo_type_classes.index(row["photo_type"])
        return img, torch.tensor(model_label), torch.tensor(type_label)


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


# Параметри
boat_model_classes = df["boat_model"].unique().tolist()
photo_type_classes = ["interior", "exterior"]
batch_size = 16
max_epochs = 50  # максимальна кількість епох
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Dataset & split
dataset = BoatDataset("boat_dataset.csv", boat_model_classes, photo_type_classes, transform)
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4)

# Модель
model = BoatClassifier(num_boat_models=len(boat_model_classes), num_photo_types=2)
model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float("inf")
best_model_path = "best_boat_classifier.pth"

for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} Train", leave=False)

    for imgs, model_labels, type_labels in loop:
        imgs = imgs.to(device)
        model_labels = model_labels.to(device)
        type_labels = type_labels.to(device)

        optimizer.zero_grad()
        model_logits, type_logits = model(imgs)
        loss_model = criterion(model_logits, model_labels)
        loss_type = criterion(type_logits, type_labels)
        loss = loss_model + loss_type
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        loop.set_postfix(loss=(running_loss / ((loop.n + 1) * batch_size)))

    avg_train_loss = running_loss / len(train_idx)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, model_labels, type_labels in val_loader:
            imgs = imgs.to(device)
            model_labels = model_labels.to(device)
            type_labels = type_labels.to(device)
            model_logits, type_logits = model(imgs)
            loss_model = criterion(model_logits, model_labels)
            loss_type = criterion(type_logits, type_labels)
            val_loss += (loss_model + loss_type).item() * imgs.size(0)

    avg_val_loss = val_loss / len(val_idx)

    print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Збереження найкращої моделі
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

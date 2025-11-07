import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "boat_dataset_3_class1.csv"
best_model_path = "best_photo_type_classifier1.pth"
max_epochs = 25


class BoatPhotoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["photo_type"].isin(["in", "out", "boat"])].reset_index(drop=True)
        self.classes = sorted(self.df["photo_type"].unique())  # ['boat', 'in', 'out']
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(row["photo_type"])
        return img, torch.tensor(label)


class PhotoTypeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = BoatPhotoDataset(csv_path, transform)

train_idx, val_idx = train_test_split(
    range(len(dataset)), test_size=0.1, random_state=42, stratify=dataset.df["photo_type"]
)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhotoTypeClassifier(num_classes=len(dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_val_loss = float("inf")



for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total * 100

    print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

print("\nTraining finished ✅")
print(f"Best model saved to {best_model_path}")

# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from base.constants import MAX_EPOCHS, BATCH_SIZE


def train_base_model(device, dataset, model, target_name, best_model_path):
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=42, stratify=dataset.df[target_name]
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Init model
    model.to(device)
    # Train Configs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")
    # Start training Loop
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]", leave=False)
    
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
    
        # Validation:
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
    
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
        # Save best model:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
    
    print("\nTraining finished ✅")
    print(f"Best model saved to {best_model_path}")



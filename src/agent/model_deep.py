import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def data_loader(images_path):

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(images_path / "train", transform=train_tf)
    val_ds = ImageFolder(images_path / "val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=16)

    print(f"Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs")
    print(f"Clases: {train_ds.classes}")

    return train_loader, val_loader


def model_definition(device):
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def train_epoch(device, model, criterion,  loader, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (model(imgs).argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def val_epoch(device, model, loader):
    model.eval()
    correct, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / len(loader.dataset), all_preds, all_labels


if __name__ == "__main__":
    # IMAGES_PROCESSED = Path("../../data/processed/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    # train_loader, val_loader = data_loader(IMAGES_PROCESSED)
    model, criterion = model_definition(device)
    print("Model already")

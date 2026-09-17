import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from models import get_model

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

class ThaiCharacterDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label = int(row['class_idx'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for damaged image
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size=224):
    """
    Returns train and validation transforms tailored for Thai character OCR.
    Crucial design choice: NO Horizontal or Vertical Flips!
    Flipping changes Thai characters semantically (e.g., ด vs ค, ภ vs ถ).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=(-12, 12), fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.94, 1.06),
            shear=(-8, 8),
            fill=255
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def calculate_class_weights(df_train, num_classes, device):
    """
    Compute inverse square root class frequencies to counter severe class imbalance
    (e.g., class with 1 sample vs class with 4,863 samples).
    """
    counts = np.zeros(num_classes)
    class_counts = df_train['class_idx'].value_counts()
    for idx, count in class_counts.items():
        counts[int(idx)] = count
        
    counts = np.maximum(counts, 1.0)
    # Square root smoothing prevents extreme dominating weights
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            batch_acc = (batch_correct / labels.size(0)) * 100.0
            print(f"  --> Step [{batch_idx + 1:>4}/{total_batches}] | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:>5.1f}% | Running Acc: {(correct/total)*100:>5.1f}%", flush=True)
            
    if scheduler is not None:
        scheduler.step()
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        
        # Top-1
        _, preds = torch.max(outputs, 1)
        correct_top1 += (preds == labels).sum().item()
        
        # Top-3
        _, top3_preds = outputs.topk(min(3, outputs.size(1)), 1, True, True)
        correct_top3 += top3_preds.eq(labels.view(-1, 1).expand_as(top3_preds)).sum().item()
        
        total += labels.size(0)
        
    val_loss = running_loss / total
    top1_acc = (correct_top1 / total) * 100.0
    top3_acc = (correct_top3 / total) * 100.0
    return val_loss, top1_acc, top3_acc

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Thai Character Classification")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--metadata", type=str, default="dataset_metadata.csv", help="Path to metadata csv")
    parser.add_argument("--save_path", type=str, default="best_model.pt", help="Path to save best weights")
    parser.add_argument("--use_weighted_loss", action="store_true", default=True, help="Use weighted cross-entropy for class imbalance")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Thai Character CNN Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model: {args.model} (Transfer Learning)")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    
    # Check metadata
    if not os.path.exists(args.metadata):
        print(f"Metadata file {args.metadata} not found. Running prepare_dataset.py...")
        from prepare_dataset import prepare_data
        prepare_data()
        
    df = pd.read_csv(args.metadata)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    with open("char_mapping.json", "r", encoding="utf-8") as f:
        char_mapping = json.load(f)
        
    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")
    
    # Transforms & Datasets
    train_transform, val_transform = get_transforms(img_size=args.img_size)
    train_dataset = ThaiCharacterDataset(df_train, transform=train_transform)
    val_dataset = ThaiCharacterDataset(df_val, transform=val_transform)
    
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    # Model
    model = get_model(model_name=args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss function
    if args.use_weighted_loss:
        class_weights = calculate_class_weights(df_train, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted Cross-Entropy Loss to counter class imbalance.")
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Training Loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_top1_acc": [], "val_top3_acc": []}
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_top1_acc"].append(val_top1)
        history["val_top3_acc"].append(val_top3)
        
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_time:.1f}s) - LR: {lr_current:.2e} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-3: {val_top3:.2f}%", end="")
        
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            checkpoint = {
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "classes": classes,
                "char_mapping": char_mapping,
                "val_top1_acc": val_top1,
                "val_top3_acc": val_top3,
                "img_size": args.img_size,
                "history": history
            }
            torch.save(checkpoint, args.save_path)
            print(" -> [BEST SAVED]")
        else:
            print()
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved best model weights to: {args.save_path}")
    
    # Save training curves plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Val Loss", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), history["train_acc"], label="Train Acc", color="blue", linewidth=2)
    plt.plot(range(1, args.epochs + 1), history["val_top1_acc"], label="Val Top-1 Acc", color="green", linewidth=2)
    plt.plot(range(1, args.epochs + 1), history["val_top3_acc"], label="Val Top-3 Acc", color="orange", linestyle="--", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curves (Best Val: {best_val_acc:.2f}%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_file = "training_curves.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved training curves to {plot_file}")
    
    # Save training summary JSON
    summary = {
        "model_name": args.model,
        "best_val_top1_acc": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "total_training_time_min": total_time / 60.0,
        "history": history
    }
    with open("training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved training_summary.json")

if __name__ == "__main__":
    main()

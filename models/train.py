import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Enable maximum CPU parallelism
torch.set_num_threads(os.cpu_count() or 8)

class BengaliNet222(nn.Module):
    def __init__(self, num_classes=222):
        super(BengaliNet222, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 (64x64 -> 32x32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2: 32 -> 64 (32x32 -> 16x16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            # Block 3: 64 -> 128 (16x16 -> 8x8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4: 128 -> 256 (8x8 -> 4x4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FastBengaliDataset(Dataset):
    def __init__(self, root_dir="DATASET", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        with open(self.root_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.class_to_idx = {}
        folders = sorted([f for f in self.root_dir.iterdir() if f.is_dir()])
        for idx, folder in enumerate(folders):
            self.class_to_idx[folder.name] = idx
            for img_path in folder.glob("*.png"):
                self.samples.append((str(img_path), idx))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(epochs=50, batch_size=128, lr=2.5e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training BengaliNet222 on device: {device} (CPU Threads: {torch.get_num_threads()})")
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = FastBengaliDataset(root_dir="DATASET", transform=train_transform)
    num_total = len(full_dataset)
    num_val = int(0.15 * num_total)
    num_train = num_total - num_val
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [num_train, num_val], generator=generator)
    val_ds.dataset.transform = val_transform
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset Loaded: Total = {num_total} | Train = {num_train} | Validation = {num_val}")
    
    model = BengaliNet222(num_classes=222).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_acc = 0.0
    start_epoch = 1
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "bengali_char_net_best.pth"
    
    if ckpt_path.exists():
        print(f"Resuming training from checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        print(f"Loaded Checkpoint: Resuming from Epoch {start_epoch} (Best Val Acc: {best_val_acc:.2f}%)", flush=True)

    for group in optimizer.param_groups:
        group['initial_lr'] = lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch - 1 if start_epoch > 1 else -1)
    
    with open(models_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)
        
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            total_train += images.size(0)
            
        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct / total_train) * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                total_val += images.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct / total_val) * 100.0
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%", flush=True)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'class_to_idx': full_dataset.class_to_idx
            }, models_dir / "bengali_char_net_best.pth")
            
    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed/60:.2f} mins!", flush=True)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)

if __name__ == "__main__":
    train_model(epochs=50, batch_size=128, lr=2.5e-3)

# Day 2: Logging annotations

# Day 2: Fallback notes

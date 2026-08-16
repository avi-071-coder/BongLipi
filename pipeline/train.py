"""
=============================================================================
BONGLIPI - Optimized Dataset & Training (Single File)
=============================================================================
Pre-loads ALL images into memory for fast training on CPU.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import time
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# SQUEEZE-AND-EXCITATION BLOCK
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch//r, 8)), nn.ReLU(inplace=True),
            nn.Linear(max(ch//r, 8), ch), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.pool(x).flatten(1)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity())
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.skip(x), inplace=True)


# ============================================================================
# BONGNET LITE MODEL
# ============================================================================

class BongNetLite(nn.Module):
    """~2.8M params, optimized for CPU training on 64x64 grayscale."""
    
    def __init__(self, num_classes=72, drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            # Stage 1: 64→32
            ResBlock(16, 32, 2), ResBlock(32, 32),
            # Stage 2: 32→16
            ResBlock(32, 64, 2), ResBlock(64, 64),
            # Stage 3: 16→8
            ResBlock(64, 128, 2), ResBlock(128, 128),
            # Stage 4: 8→4
            ResBlock(128, 256, 2), ResBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop), nn.Linear(256, num_classes)
        )
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.head(self.features(x))
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MEMORY-CACHED DATASET
# ============================================================================

class CachedBengaliDataset(Dataset):
    """
    Loads ALL images into RAM at init time for maximum training speed.
    Images are stored as uint8 numpy arrays, converted to tensors on-the-fly.
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load label map
        with open(self.data_dir / "label_map.json", "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.num_classes = int(self.label_map["num_classes"])
        
        # Pre-load all images into memory
        print("Loading dataset into memory...", end=" ", flush=True)
        t0 = time.time()
        
        self.images = []  # list of np.uint8 arrays (64x64)
        self.labels = []  # list of int labels
        
        for cls_id in range(self.num_classes):
            cls_dir = self.data_dir / f"class_{cls_id:03d}"
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.images.append(img)
                    self.labels.append(cls_id)
        
        print(f"Done! {len(self.images)} samples in {time.time()-t0:.1f}s")
        
        # Convert to numpy arrays for faster access
        self.images = np.array(self.images, dtype=np.uint8)  # (N, 64, 64)
        self.labels = np.array(self.labels, dtype=np.int64)  # (N,)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]  # uint8 numpy (64, 64)
        label = int(self.labels[idx])
        
        # Convert to PIL for transforms
        img = Image.fromarray(img, mode='L')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        
        return img, label


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=4),
            T.RandomPerspective(distortion_scale=0.08, p=0.3),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])


# ============================================================================
# TRAINING
# ============================================================================

def train():
    CONFIG = {
        "data_dir": "dataset",
        "batch_size": 64,
        "epochs": 50,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "dropout": 0.25,
        "label_smoothing": 0.1,
        "patience": 12,
    }
    
    OUT = Path("checkpoints")
    OUT.mkdir(exist_ok=True)
    
    # ---- Load data ----
    full_data = CachedBengaliDataset(CONFIG["data_dir"])
    n = len(full_data)
    n_test = int(n * 0.10)
    n_val = int(n * 0.15)
    n_train = n - n_val - n_test
    
    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=gen)
    
    train_imgs = full_data.images[train_idx.indices]
    train_lbls = full_data.labels[train_idx.indices]
    val_imgs = full_data.images[val_idx.indices]
    val_lbls = full_data.labels[val_idx.indices]
    test_imgs = full_data.images[test_idx.indices]
    test_lbls = full_data.labels[test_idx.indices]
    
    class SplitDataset(Dataset):
        def __init__(self, imgs, lbls, transform):
            self.imgs = imgs
            self.lbls = lbls
            self.transform = transform
        def __len__(self):
            return len(self.lbls)
        def __getitem__(self, idx):
            img = Image.fromarray(self.imgs[idx], mode='L')
            if self.transform:
                img = self.transform(img)
            return img, int(self.lbls[idx])
    
    train_loader = DataLoader(SplitDataset(train_imgs, train_lbls, get_transforms(True)),
                              batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(SplitDataset(val_imgs, val_lbls, get_transforms(False)),
                            batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(SplitDataset(test_imgs, test_lbls, get_transforms(False)),
                             batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    
    # ---- Model ----
    model = BongNetLite(num_classes=full_data.num_classes, drop=CONFIG["dropout"])
    print(f"Model: BongNetLite ({model.count_params():,} params)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr"], epochs=CONFIG["epochs"],
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )
    
    # ---- Training loop ----
    best_val_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0., 0, 0
        for imgs, lbls in train_loader:
            out = model(imgs)
            loss = criterion(out, lbls)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += out.argmax(1).eq(lbls).sum().item()
            train_total += imgs.size(0)
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # VALIDATE
        model.eval()
        val_loss, val_correct, val_total = 0., 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                val_correct += out.argmax(1).eq(lbls).sum().item()
                val_total += imgs.size(0)
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        marker = ""
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "num_classes": full_data.num_classes, "label_map": full_data.label_map,
                "best_val_acc": best_val_acc, "config": CONFIG,
            }, OUT / "best_model.pth")
            marker = f" ★ (+{improvement:.1f}%)"
        else:
            patience_ctr += 1
        
        lr = optimizer.param_groups[0]['lr']
        patience_str = "" if patience_ctr == 0 else f" [{patience_ctr}/{CONFIG['patience']}]"
        print(f"E{epoch:02d} {dt:.0f}s | Train {train_acc:.1f}% ({train_loss:.3f}) | "
              f"Val {val_acc:.1f}% ({val_loss:.3f}) | LR {lr:.5f}{marker}{patience_str}")
        
        if patience_ctr >= CONFIG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # ---- TEST ----
    best = torch.load(OUT / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out = model(imgs)
            preds = out.argmax(1)
            test_correct += preds.eq(lbls).sum().item()
            test_total += imgs.size(0)
            all_preds.extend(preds.numpy())
            all_labels.extend(lbls.numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    # Per-class accuracy
    cls_correct = defaultdict(int)
    cls_total = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        cls_total[l] += 1
        if p == l: cls_correct[l] += 1
    
    accs = [(cid, full_data.label_map["id_to_char"].get(str(cid), "?"),
             100.*cls_correct[cid]/cls_total[cid] if cls_total[cid] else 0,
             cls_total[cid]) for cid in sorted(cls_total.keys())]
    accs.sort(key=lambda x: x[2])
    
    print("\nHardest classes:")
    for cid, char, acc, tot in accs[:10]:
        print(f"  {char} (class_{cid:03d}): {acc:.1f}% ({tot} samples)")
    
    print("\nEasiest classes:")
    for cid, char, acc, tot in accs[-5:]:
        print(f"  {char} (class_{cid:03d}): {acc:.1f}% ({tot} samples)")
    
    # Save everything
    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    results = {"test_accuracy": test_acc, "best_val_accuracy": best_val_acc,
               "epochs_trained": epoch, "total_params": model.count_params()}
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nModel & results saved to {OUT}/")


if __name__ == "__main__":
    train()

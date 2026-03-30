#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, glob, random

PATCH_SIZE = 64
LINE_WIDTH_RANGE = (4, 9)
SHIFT_RANGE = (0.5, 6.0)
N_TRAIN = 8000
N_VAL = 1500
EPOCHS = 40
BATCH = 32
LR = 1e-3


def load_noise_patches(directories, n_patches=500, patch_size=80):
    patches = []
    for d in directories:
        tiles = sorted(glob.glob(os.path.join(d, "TC*.jpg")))[:20]
        for tile_path in tiles:
            gray = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h, w = gray.shape
            for _ in range(n_patches // len(tiles) // len(directories) + 1):
                x = random.randint(50, w - patch_size - 50)
                y = random.randint(50, min(h // 3, h - patch_size - 50))
                patch = gray[y:y + patch_size, x:x + patch_size].astype(np.float32)
                if np.std(patch) < 25:
                    patches.append(patch)
    print(f"Loaded {len(patches)} noise patches from {len(directories)} directories")
    return patches


def draw_line(img, pos, direction, width, contrast):
    h, w = img.shape
    sigma = width / 3
    if direction == 'h':
        for y in range(h):
            val = contrast * np.exp(-0.5 * ((y - pos) / sigma) ** 2)
            img[y, :] -= val
    else:
        for x in range(w):
            val = contrast * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            img[:, x] -= val


def generate_displacement_patch(noise_patches, shift_px):
    size = PATCH_SIZE
    bg = random.choice(noise_patches)
    bg = cv2.resize(bg, (size, size))
    img = bg.copy()

    direction = random.choice(['h', 'v'])
    line_width = random.uniform(*LINE_WIDTH_RANGE)
    contrast = random.uniform(8, 40)
    center = size // 2 + random.uniform(-8, 8)

    if shift_px > 0:
        split = random.randint(size // 4, 3 * size // 4)
        sigma = line_width / 3

        if direction == 'h':
            for x in range(size):
                pos = center if x < split else center + shift_px * random.choice([-1, 1])
                for y in range(size):
                    val = contrast * np.exp(-0.5 * ((y - pos) / sigma) ** 2)
                    img[y, x] -= val
        else:
            for y in range(size):
                pos = center if y < split else center + shift_px * random.choice([-1, 1])
                for x in range(size):
                    val = contrast * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
                    img[y, x] -= val
    else:
        draw_line(img, center, direction, line_width, contrast)

    img = np.clip(img, 0, 255)

    if random.random() < 0.5:
        img = np.fliplr(img).copy()
    if random.random() < 0.5:
        img = np.flipud(img).copy()
    if random.random() < 0.3:
        img = np.rot90(img, random.randint(1, 3)).copy()

    return img.astype(np.float32)


def generate_stray_mark_patch(noise_patches):
    size = PATCH_SIZE
    bg = random.choice(noise_patches)
    bg = cv2.resize(bg, (size, size))
    img = bg.copy()

    direction = random.choice(['h', 'v'])
    line_width = random.uniform(*LINE_WIDTH_RANGE)
    contrast = random.uniform(8, 40)
    center = size // 2 + random.uniform(-8, 8)

    draw_line(img, center, direction, line_width, contrast)

    mark_type = random.choice(['dot', 'stroke'])
    offset = random.uniform(8, 20) * random.choice([-1, 1])

    if direction == 'h':
        mx = random.randint(10, size - 10)
        my = int(center + offset)
    else:
        mx = int(center + offset)
        my = random.randint(10, size - 10)

    if mark_type == 'dot':
        r = random.randint(2, 5)
        cv2.circle(img, (mx, my), r, float(img[my, mx] - contrast * 0.7), -1)
    else:
        length = random.randint(5, 15)
        angle = random.uniform(0, np.pi)
        dx = int(length * np.cos(angle))
        dy = int(length * np.sin(angle))
        cv2.line(img, (mx, my), (mx + dx, my + dy),
                 float(img[my, mx] - contrast * 0.5), max(1, int(line_width * 0.5)))

    img = np.clip(img, 0, 255)
    return img.astype(np.float32)


class ShiftDataset(Dataset):
    def __init__(self, noise_patches, n_samples):
        self.noise = noise_patches
        self.data = []
        self._generate(n_samples)

    def _generate(self, n):
        for _ in range(n):
            r = random.random()
            if r < 0.15:
                patch = generate_stray_mark_patch(self.noise)
                label = 0.0
            elif r < 0.30:
                patch = generate_displacement_patch(self.noise, 0)
                label = 0.0
            else:
                shift = random.uniform(*SHIFT_RANGE)
                patch = generate_displacement_patch(self.noise, shift)
                label = shift
            self.data.append((patch, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch, label = self.data[idx]
        tensor = torch.from_numpy(patch / 255.0).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.float32)


class ShiftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x).squeeze(-1)


def train():
    print("Loading noise textures from real tiles...")
    dirs = [d for d in ["grid_test_elitho_2026-01-29", "grid_test_2"] if os.path.isdir(d)]
    noise_patches = load_noise_patches(dirs)

    if len(noise_patches) < 50:
        print("Not enough noise patches, generating synthetic noise")
        for _ in range(200):
            noise_patches.append(np.random.normal(130, 10, (80, 80)).astype(np.float32))

    print("Generating synthetic datasets...")
    train_ds = ShiftDataset(noise_patches, N_TRAIN)
    val_ds = ShiftDataset(noise_patches, N_VAL)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = ShiftNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for patches, labels in train_dl:
            patches, labels = patches.to(device), labels.to(device)
            preds = model(patches)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(patches)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for patches, labels in val_dl:
                patches, labels = patches.to(device), labels.to(device)
                preds = model(patches)
                val_loss += loss_fn(preds, labels).item() * len(patches)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        mask = labels_arr > 0
        mae_shift = np.mean(np.abs(preds_arr[mask] - labels_arr[mask])) if np.sum(mask) > 0 else 0

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS}: train={train_loss:.4f} "
                  f"val={val_loss:.4f} MAE={mae_shift:.2f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "shift_model.pt")

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print("Saved: shift_model.pt")


if __name__ == "__main__":
    train()

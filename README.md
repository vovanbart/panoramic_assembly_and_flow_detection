# Panoramic Assembly and Defect Detection

Automated stitching of 100 microscope images (10×10 snake grid) of lithographic beam-printed structures with YOLO-based defect detection.

## What it does

1. **Stitching** — assembles 100 overlapping tile images into a single mosaic using SIFT matching, phase correlation, template matching, and global least-squares alignment
2. **Defect detection** — finds manufacturing defects (displaced lines, black spots) on the stitched grid using YOLOv8 and measures displacement with a CNN (ShiftNet)

## Usage

```bash
python3 stitch.py grid_test_elitho_2026-01-29
```

Outputs:
- `*_stitched.png` / `*_stitched.jpg` — mosaic
- `*_defects.jpg` — mosaic with annotated defects
- `*_defects.json` — defect coordinates and measurements

## Training

```bash
python3 train_yolo.py --epochs 100 --data dataset_yolo/data.yaml
python3 train_shift_model.py
```

## Requirements

```bash
pip install opencv-python numpy scipy torch ultralytics
```

## Files

| File | Description |
|------|-------------|
| `stitch.py` | Main pipeline: stitching + detection |
| `train_yolo.py` | YOLOv8 training on defect dataset |
| `train_shift_model.py` | ShiftNet training on synthetic data |
| `best.pt` | Trained YOLO weights |
| `shift_model.pt` | Trained ShiftNet weights |

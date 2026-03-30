#!/usr/bin/env python3

from ultralytics import YOLO
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--data", default="dataset_yolo/data.yaml")
    a = ap.parse_args()

    model = YOLO(a.model)

    results = model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device="mps",
        patience=30,
        save=True,
        plots=True,
        augment=True,
        workers=2,
        cache=False,
        flipud=0.5,
        fliplr=0.5,
        degrees=5.0,
        scale=0.3,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
    )

    print(f"\nBest model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

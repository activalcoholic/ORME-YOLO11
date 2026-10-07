"""Evaluate a checkpoint against a selected YOLO dataset configuration."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--name', default='orme-evaluation')
    args = parser.parse_args()
    for path in [args.weights, args.data]:
        if not path.is_file():
            raise FileNotFoundError(path)
    from ultralytics import YOLO
    metrics = YOLO(str(args.weights.resolve())).val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str((Path('runs') / 'evaluate').resolve()), name=args.name, plots=True,
    )
    print({'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
           'mAP50': float(metrics.box.map50), 'mAP50-95': float(metrics.box.map)})


if __name__ == '__main__':
    main()

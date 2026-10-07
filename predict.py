"""Run ORME-YOLO11 image or video inference."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    from ultralytics import YOLO
    for _ in YOLO(str(args.weights.resolve())).predict(
        source=args.source, device=args.device, imgsz=args.imgsz,
        conf=args.conf, save=True, stream=True,
        project=str((Path('runs') / 'predict').resolve()), name='orme',
    ):
        pass


if __name__ == '__main__':
    main()

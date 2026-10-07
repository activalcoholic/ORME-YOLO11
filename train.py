"""Train ORME-YOLO11 with explicit custom loss and augmentation settings."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'yolo11n-mbfem.yaml')
    parser.add_argument('--pretrained', default='yolo11n.pt', help="Path or Ultralytics model name; 'none' trains from scratch")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='orme-yolo11n')
    parser.add_argument('--bbox-loss', choices=['ciou', 'mpdiou', 'focaler_ciou', 'siou'], default='mpdiou')
    parser.add_argument('--mpdiou-weight', type=float, default=1.0)
    parser.add_argument('--rep-alpha', type=float, default=0.0001)
    parser.add_argument('--rep-beta', type=float, default=0.0)
    parser.add_argument('--underwater-p', type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.underwater_p <= 1:
        parser.error('--underwater-p must be in [0, 1]')
    if args.rep_alpha < 0 or args.rep_beta < 0:
        parser.error('Repulsion loss weights must be nonnegative')
    return args


def remap_detect_pretrained_weights(custom_model, weights_path):
    """Transfer matching Detect tensors despite the MBFEM layer index shift."""
    from ultralytics import YOLO
    base_model = YOLO(weights_path)
    source_state = base_model.model.float().state_dict()
    target_state = custom_model.model.state_dict()
    source_prefix = f'model.{len(base_model.model.model) - 1}.'
    target_prefix = f'model.{len(custom_model.model.model) - 1}.'
    remapped = {}
    for key, value in source_state.items():
        if key.startswith(source_prefix):
            target_key = target_prefix + key[len(source_prefix):]
            if target_key in target_state and value.shape == target_state[target_key].shape:
                remapped[target_key] = value
    custom_model.model.load_state_dict(remapped, strict=False)
    print(f'Detect remap: {source_prefix} -> {target_prefix}; {len(remapped)} tensors')
    return custom_model


def main():
    args = parse_args()
    for path in [args.data, args.model]:
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = {
        'BBOX_LOSS_TYPE': args.bbox_loss,
        'MPDIOU_WEIGHT': str(args.mpdiou_weight),
        'USE_REPLOSS': str(int(args.rep_alpha > 0 or args.rep_beta > 0)),
        'REP_ALPHA': str(args.rep_alpha), 'REP_BETA': str(args.rep_beta),
        'REP_PNMS': '0.45', 'REP_GTNMS': '0.4', 'REP_MARGIN': '0.05',
        'UNDERWATER_DEGRADE_P': str(args.underwater_p),
        'TAL_TOPK': '10', 'TAL_TOPK2': '10', 'CLS_POS_WEIGHT': '1.0',
        'FOCALER_LOWER': '0.0', 'FOCALER_UPPER': '0.95',
    }
    os.environ.update(settings)
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(args.seed, deterministic=True)
    model = YOLO(str(args.model.resolve()))
    pretrained = args.pretrained.lower() != 'none'
    if pretrained:
        model.load(args.pretrained)
        remap_detect_pretrained_weights(model, args.pretrained)

    def save_custom_settings(trainer):
        output = Path(trainer.save_dir) / 'orme_settings.json'
        output.write_text(json.dumps(settings, indent=2) + '\n')

    model.add_callback('on_train_start', save_custom_settings)
    model.train(
        data=str(args.data.resolve()), seed=args.seed, deterministic=True,
        pretrained=args.pretrained if pretrained else False,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        device=args.device, project=str(Path(args.project).resolve()), name=args.name,
        optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, mosaic=1.0, mixup=0.1, close_mosaic=5,
        degrees=5.0, translate=0.1, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,
    )
    print(f'Best weights: {model.trainer.best}')


if __name__ == '__main__':
    main()

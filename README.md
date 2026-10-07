# ORME-YOLO11

ORME-YOLO11 is an underwater fish detector for Low activity, Routine swimming, and Feeding aggregation scenarios. The model is based on YOLO11, features the MBFEM module, and includes MPDIoU, RepGT, and underwater degradation augmentation.

## Installation

Python 3.10 is recommended. First install PyTorch and torchvision suitable for your device, then run:

```bash
git clone https://github.com/activalcoholic/ORME-YOLO11.git
cd ORME-YOLO11
python -m pip install -e .
```


## Dataset

Download **the Cross-Scenario Underwater Fish Dataset**: [GitHub Release](https://github.com/activalcoholic/ORME-YOLO11/releases/tag/cross-scenario-underwater-fish-dataset).

| File | Contents |
| --- | --- |
| [Dataset-train.zip](https://github.com/activalcoholic/ORME-YOLO11/releases/download/cross-scenario-underwater-fish-dataset/the-Cross-Scenario-Underwater-Fish-Dataset-train.zip) | Training set: 3980 images and their labels |
| [Dataset-val.zip](https://github.com/activalcoholic/ORME-YOLO11/releases/download/cross-scenario-underwater-fish-dataset/the-Cross-Scenario-Underwater-Fish-Dataset-val.zip) | Validation set: 740 images and their labels |
| [Dataset-test.zip](https://github.com/activalcoholic/ORME-YOLO11/releases/download/cross-scenario-underwater-fish-dataset/the-Cross-Scenario-Underwater-Fish-Dataset-test.zip) | Test set: three scenarios, each with 200 images and their labels |

Extract all three archives into the repository's `data/` directory. Each scenario folder in the test set contains `images` and `labels`:

```text
data/the Cross-Scenario Underwater Fish Dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── test/
│   ├── Low activity/
│   │   ├── images/
│   │   └── labels/
│   ├── Routine swimming/
│   │   ├── images/
│   │   └── labels/
│   └── Feeding aggregation/
│       ├── images/
│       └── labels/
├── data.yaml
├── data_low_activity.yaml
├── data_routine_swimming.yaml
└── data_feeding_aggregation.yaml
```

Labels use the YOLO format. The four dataset configuration files in the repository root can be used directly; the archives also include configurations with paths relative to the dataset directory.

## Training

```bash
python train.py --data data.yaml --device 0
```

The default model configuration is [`yolo11n-mbfem.yaml`](yolo11n-mbfem.yaml). The default training settings are 100 epochs, an image size of 640, a batch size of 32, the AdamW optimizer, and a random seed of 0.

```bash
# Reduce the batch size if GPU memory is insufficient
python train.py --data data.yaml --device 0 --batch 8
```

Training outputs are saved in `runs/detect/`, including the best weights and the actual training settings.

## Evaluation and Inference

Replace `--weights` with the path to the `best.pt` file generated during training:

```bash
# Full test set
python evaluate.py --weights path/to/best.pt --data data.yaml

# Evaluate the three scenarios separately
python evaluate.py --weights path/to/best.pt --data data_low_activity.yaml --name low_activity
python evaluate.py --weights path/to/best.pt --data data_routine_swimming.yaml --name routine_swimming
python evaluate.py --weights path/to/best.pt --data data_feeding_aggregation.yaml --name feeding_aggregation

# Image or video inference
python predict.py --weights path/to/best.pt --source path/to/image.jpg --device 0
```

## Code Locations

| Function | File |
| --- | --- |
| MBFEM | `ultralytics/nn/modules/MBFEM.py` |
| Module imports and model parsing | `ultralytics/nn/modules/__init__.py`, `ultralytics/nn/tasks.py` |
| MPDIoU, RepGT | `ultralytics/utils/loss.py` |
| Underwater degradation augmentation | `ultralytics/data/augment.py` |
| Model configuration | `yolo11n-mbfem.yaml` |

## License

The code retains the AGPL-3.0 license of the original Ultralytics project. See [`LICENSE`](LICENSE).


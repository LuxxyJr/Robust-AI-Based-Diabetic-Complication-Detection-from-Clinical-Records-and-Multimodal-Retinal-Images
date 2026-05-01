# Track 2 Retinal Imaging Pipeline

Reproducible baseline pipeline for the MMRDR retinal image dataset.

The pipeline keeps the official dataset split encoded in image names:

- `tr*.jpg` is used for training and validation.
- `ts*.jpg` is used only for locked testing.

## Recommended Order

```bash
python src/audit_dataset.py --data-root "../Datasets/Paper 2"
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model resnet50
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task cfp_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task uwf_dr --model mobilenet_v3
python src/train_retinal_baseline.py --data-root "../Datasets/Paper 2" --task oct_dme --model mobilenet_v3
```

## Outputs

- Dataset audit: `outputs/dataset_audit/`
- Model runs: `outputs/<task>/<model>/`
- Each model run saves metrics, confusion matrix, best checkpoint, predictions, and Grad-CAM examples.

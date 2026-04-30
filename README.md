# African Landmark Recognition

ICS555 Computer Vision Capstone — Ashesi University, Semester 2, 2026.

A systematic study comparing CNN-from-scratch and fine-tuned pretrained architectures for landmark classification, with a critical ethical analysis of African landmark underrepresentation in public vision datasets.

---

## Results Summary

| Run | Architecture | Strategy | Augmentation | Epochs | Top-1 | Top-5 | Macro-F1 |
|-----|-------------|----------|-------------|--------|-------|-------|----------|
| A0  | CustomModel (VGG-3) | scratch | full | 50 | 34.2% | 65.0% | 30.0% |
| A1  | ResNet-18 | head-only | full | 50 | 73.1% | 92.1% | 72.9% |
| A2  | ResNet-18 | full fine-tune | full | 15 | 81.1% | 93.7% | 81.2% |
| **A4** | **ResNet-50** | **full fine-tune** | **full** | **30** | **85.4%** | **94.7%** | **85.3%** |
| A6  | ViT-B/16 | head-only | full | 15 | 82.4% | 94.6% | 82.2% |
| A7  | ResNet-50 | full fine-tune | minimal | 15 | 84.3% | 95.1% | 84.3% |
| A8  | ResNet-50 | full fine-tune + WRS | full | 15 | 69.8% | 76.8% | 63.2% |

---

## Setup

```bash
git clone https://github.com/zamsi-ajegetina/landmark-recognition.git
cd african-landmark-recognition

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the landmark dataset and place it so that:
```
landmark_images/
├── train/
│   ├── class_001/
│   └── ...
└── test/
    ├── class_001/
    └── ...
```

Set the `DATA_LOCATION` environment variable or update `src/helpers.py` to point to the dataset root.

---

## Reproduce an Experiment

```bash
# Primary result (ResNet-50 full fine-tune)
python ablation_runner.py --config configs/A4_resnet50_full.yaml

# CNN from scratch baseline
python ablation_runner.py --config configs/A0_scratch.yaml

# Quick smoke test (500 images, no wandb)
python ablation_runner.py --config configs/A4_resnet50_full.yaml --limit 500 --no-wandb
```

Each run:
1. Trains for the configured number of epochs with CosineAnnealingLR
2. Saves the best checkpoint to `checkpoints/<run_name>.pt`
3. Evaluates on the test set (Top-1, Top-5, Macro-F1, per-class report)
4. Exports a TorchScript checkpoint to `checkpoints/<run_name>_exported.pt`
5. Logs all metrics to Weights & Biases

---

## Evaluate a Saved Checkpoint

```bash
# Full test-set evaluation
python evaluate.py --checkpoint checkpoints/A4_resnet50_full.pt \
                   --model-name resnet50

# Single image prediction
python evaluate.py --checkpoint checkpoints/A4_resnet50_full_exported.pt \
                   --image path/to/photo.jpg --top-k 5

# CNN from scratch
python evaluate.py --checkpoint checkpoints/A0_scratch.pt --scratch
```

---

## Gradio Demo

```bash
CHECKPOINT_PATH=checkpoints/A4_resnet50_full_exported.pt \
RAW_CHECKPOINT=checkpoints/A4_resnet50_full.pt \
python app/gradio_app.py
```

Open `http://localhost:7860`. Optionally set `GEMINI_API_KEY` for Gemini-powered landmark enrichment.

---

## Google Colab

Training notebooks:
- `notebooks/02_cnn_from_scratch.ipynb` — A0 baseline
- `notebooks/03_transfer_learning.ipynb` — A1–A2 experiments
- `notebooks/04_ablation_study.ipynb` — A4, A6, A7, A8

Quick start in Colab:
```python
!git clone https://github.com/zamsi-ajegetina/landmark-recognition.git
%cd african-landmark-recognition
!pip install -r requirements.txt
!python ablation_runner.py --config configs/A4_resnet50_full.yaml
```

---

## Project Structure

```
african-landmark-recognition/
├── configs/              # YAML experiment configs (one per ablation run)
├── src/
│   ├── data.py           # Data loaders with augmentation + WeightedRandomSampler
│   ├── model.py          # CNN from scratch (CustomModel)
│   ├── transfer.py       # Pretrained model setup (ResNet, ViT, EfficientNet)
│   ├── optimization.py   # Loss (with label smoothing) and optimizer factory
│   ├── train.py          # Training loop with wandb logging + Top-5 metrics
│   ├── predictor.py      # TorchScript-compatible predictor class
│   └── helpers.py        # Dataset stats, visualization utilities
├── notebooks/            # Colab-ready experiment notebooks
├── app/
│   └── gradio_app.py     # Web demo with GradCAM and optional Gemini enrichment
├── ablation_runner.py    # Config-driven experiment runner
├── evaluate.py           # Standalone evaluation (Top-1, Top-5, Macro-F1)
└── requirements.txt
```

---

## Model Weights

Pre-trained checkpoints are available on HuggingFace Hub:
[Checkpoints](https://drive.google.com/drive/folders/1S80KaXhe-aty3uliVmSe7dZ8C2sj-K49?usp=sharing)

---

## Acknowledgements

- Dataset: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
- Pretrained weights: [torchvision ImageNet-1K](https://pytorch.org/vision/stable/models.html)
- Experiment tracking: [Weights & Biases](https://wandb.ai)
- GradCAM: [torchcam](https://github.com/frgfm/torch-cam)

This project was developed with assistance from Claude(Anthropic).
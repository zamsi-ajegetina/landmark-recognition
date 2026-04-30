# African Landmark Recognition

ICS555 Computer Vision Capstone — Ashesi University, Semester 2, 2026.

A systematic study comparing CNN-from-scratch and fine-tuned pretrained architectures for landmark classification, with a critical ethical analysis of African landmark underrepresentation in public vision datasets.

**Team:** Atsu Jegetina · Nana Yaw Adjei Koranteng · Takyi Kevin Yeboah · Abubakari Sadik Osman

---

## Results Summary

| Run | Architecture | Strategy | Augmentation | Epochs | Top-1 | Top-5 | Macro-F1 |
|-----|-------------|----------|-------------|--------|-------|-------|----------|
| A0  | CustomModel (VGG-style) | scratch | full | 50 | 34.2% | 65.0% | 30.0% |
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

**Google Colab:** the dataset is downloaded automatically when you run the ablation or GradCAM notebooks — no manual steps needed.

**Local setup:** download the dataset from Google Drive and extract it at the project root:

**[Download Dataset](https://drive.google.com/file/d/10hLpehomjhFJTNbJkCRyc4lkV4Iaouan/view?usp=sharing)**

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

## Notebooks (Google Colab)

Open these notebooks in Google Colab for GPU-accelerated training and analysis.

### Run Ablation Experiments

`notebooks/ablation_study.ipynb` — trains all seven ablation runs (A0–A8) using the config-driven runner. Each cell runs one experiment and logs metrics to Weights & Biases.

```python
# Inside the notebook (Colab)
!git clone https://github.com/zamsi-ajegetina/landmark-recognition.git
%cd african-landmark-recognition
!pip install -r requirements.txt

# Run a specific experiment
!python ablation_runner.py --config configs/A4_resnet50_full.yaml
```

### GradCAM Visualisation

`notebooks/gradcam_analysis.ipynb` — loads a trained checkpoint and generates GradCAM heatmaps comparing CustomModel (A0) against ResNet-50 (A4) on selected test images.

---

## Run Experiments Locally

Each config trains the model, evaluates on the test set, and exports a TorchScript checkpoint.

```bash
# Primary result — ResNet-50 full fine-tune
python ablation_runner.py --config configs/A4_resnet50_full.yaml

# CNN from scratch baseline
python ablation_runner.py --config configs/A0_scratch.yaml

# All other ablations
python ablation_runner.py --config configs/A1_resnet18_frozen.yaml
python ablation_runner.py --config configs/A2_resnet18_full.yaml
python ablation_runner.py --config configs/A6_vitb16_frozen.yaml
python ablation_runner.py --config configs/A7_resnet50_minimal_aug.yaml
python ablation_runner.py --config configs/A8_resnet50_weighted.yaml

# Quick smoke test (500 images, no W&B logging)
python ablation_runner.py --config configs/A4_resnet50_full.yaml --limit 500 --no-wandb
```

Each run:
1. Trains for the configured number of epochs with CosineAnnealingLR scheduling
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

The demo app accepts an uploaded landmark photograph and returns:
- **Top-5 predictions** with confidence scores
- **GradCAM heatmap** highlighting which image regions drove the prediction
- **Wikipedia summary** of the predicted landmark fetched automatically

```bash
# Minimal — uses auto-detected checkpoints
python app/gradio_app.py

# Explicit checkpoint paths
CHECKPOINT_PATH=checkpoints/A4_resnet50_full_exported.pt \
RAW_CHECKPOINT=checkpoints/A4_resnet50_full.pt \
python app/gradio_app.py
```

Open `http://localhost:7860` in your browser.

The app auto-detects the best available raw checkpoint for GradCAM in this priority order: A4 → A2 → A7 → A8. The TorchScript (`*_exported.pt`) checkpoint is used for fast inference.

---

## Project Structure

```
african-landmark-recognition/
├── configs/              # YAML experiment configs (one per ablation run)
│   ├── base.yaml         # Shared defaults
│   ├── A0_scratch.yaml
│   ├── A1_resnet18_frozen.yaml
│   ├── A2_resnet18_full.yaml
│   ├── A4_resnet50_full.yaml
│   ├── A6_vitb16_frozen.yaml
│   ├── A7_resnet50_minimal_aug.yaml
│   └── A8_resnet50_weighted.yaml
├── src/
│   ├── data.py           # Data loaders with augmentation + WeightedRandomSampler
│   ├── model.py          # CNN from scratch (CustomModel — VGG-style 3-block)
│   ├── transfer.py       # Pretrained model setup (ResNet-18/50, ViT-B/16)
│   ├── optimization.py   # Loss (label smoothing) and optimizer factory
│   ├── train.py          # Training loop, W&B logging, Top-1/5 metrics
│   └── helpers.py        # Dataset stats, mean/std computation
├── notebooks/
│   ├── ablation_study.ipynb    # Run all 7 ablation experiments on Colab
│   └── gradcam_analysis.ipynb  # GradCAM heatmap visualisation
├── app/
│   └── gradio_app.py     # Interactive demo (GradCAM + Wikipedia enrichment)
├── results/              # Saved figures and demo screenshots
├── paper/
│   └── main.tex          # Technical report (LaTeX)
├── ablation_runner.py    # Config-driven experiment runner
├── evaluate.py           # Standalone evaluation script
└── requirements.txt
```

---

## Model Weights

Pre-trained checkpoints are available on Google Drive:
[Download Checkpoints](https://drive.google.com/drive/folders/1S80KaXhe-aty3uliVmSe7dZ8C2sj-K49?usp=sharing)

Place downloaded `.pt` files in the `checkpoints/` directory before running evaluation or the demo.

---

## Acknowledgements

- Dataset: [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)
- Pretrained weights: [torchvision ImageNet-1K](https://pytorch.org/vision/stable/models.html)
- Experiment tracking: [Weights & Biases](https://wandb.ai)
- GradCAM: [torchcam](https://github.com/frgfm/torch-cam)

This project was developed with assistance from Claude (Anthropic).
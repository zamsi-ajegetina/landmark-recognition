"""
Standalone evaluation script.

Usage:
    # Evaluate a saved checkpoint on the test set
    python evaluate.py --checkpoint checkpoints/A4_resnet50_full.pt \\
                       --model-name resnet50 --n-classes 50

    # Evaluate the CNN-from-scratch model
    python evaluate.py --checkpoint checkpoints/A0_scratch.pt --scratch

    # Run on a single image
    python evaluate.py --checkpoint checkpoints/A4_resnet50_full_exported.pt \\
                       --image path/to/photo.jpg --top-k 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_data_loaders
from src.helpers import compute_mean_and_std
from src.model import MyModel
from src.optimization import get_loss
from src.train import one_epoch_test
from src.transfer import get_model_transfer_learning


def load_model(args) -> torch.nn.Module:
    if args.scratch:
        model = MyModel(num_classes=args.n_classes)
    else:
        model = get_model_transfer_learning(
            model_name=args.model_name,
            n_classes=args.n_classes,
            finetune_strategy="full",
        )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model


def evaluate_dataset(args):
    model = load_model(args)
    loss_fn = get_loss()
    data_loaders = get_data_loaders(batch_size=32, num_workers=2)

    print(f"Evaluating {args.checkpoint} on the test set...\n")
    test_loss, top1_acc, top5_acc, all_targets, all_preds = one_epoch_test(
        data_loaders["test"], model, loss_fn
    )

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Top-1 Accuracy : {100*top1_acc:.2f}%")
    print(f"  Top-5 Accuracy : {100*top5_acc:.2f}%")
    print(f"  Macro-F1       : {100*macro_f1:.2f}%")
    print(f"  Micro-F1       : {100*micro_f1:.2f}%")
    print(f"{'='*50}\n")

    class_names = data_loaders["test"].dataset.classes
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    # Worst 10 classes by per-class accuracy
    cm = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    worst_idx = np.argsort(per_class_acc)[:10]
    print("\nWorst 10 classes by accuracy:")
    for idx in worst_idx:
        print(f"  [{idx:2d}] {class_names[idx]:<40} {100*per_class_acc[idx]:.1f}%")


def evaluate_single_image(args):
    from PIL import Image

    mean, std = compute_mean_and_std()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img = Image.open(args.image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    # Try TorchScript first, fall back to nn.Module
    try:
        model = torch.jit.load(args.checkpoint, map_location="cpu")
    except Exception:
        model = load_model(args)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=min(args.top_k, probs.size(0)))

    data_loaders = get_data_loaders(batch_size=1, limit=1, num_workers=0)
    class_names = data_loaders["test"].dataset.classes

    print(f"\nTop-{args.top_k} predictions for {args.image}:")
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        print(f"  {prob*100:5.1f}%  {class_names[idx]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-name", default="resnet50", help="torchvision model name")
    parser.add_argument("--n-classes", type=int, default=50, help="Number of output classes")
    parser.add_argument("--scratch", action="store_true", help="Use MyModel (CNN from scratch)")
    parser.add_argument("--image", default=None, help="Single image to classify")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions for single image mode")
    args = parser.parse_args()

    if args.image:
        evaluate_single_image(args)
    else:
        evaluate_dataset(args)


if __name__ == "__main__":
    main()

"""
Config-driven ablation runner.

Usage:
    python ablation_runner.py --config configs/A4_resnet50_full.yaml
    python ablation_runner.py --config configs/A4_resnet50_full.yaml --no-wandb
    python ablation_runner.py --config configs/A0_scratch.yaml --limit 500 
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_data_loaders
from src.model import CustomModel
from src.transfer import get_model_transfer_learning
from src.optimization import get_loss, get_optimizer
from src.train import optimize, one_epoch_test

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    """Load a YAML config, merging with _base_ if specified."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_key = cfg.pop("_base_", None)
    if base_key:
        base_path = config_path.parent / base_key
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    if model_cfg["type"] == "scratch":
        return CustomModel(num_classes=model_cfg["n_classes"])
    elif model_cfg["type"] == "transfer":
        return get_model_transfer_learning(
            model_name=model_cfg["model_name"],
            n_classes=model_cfg["n_classes"],
            finetune_strategy=model_cfg.get("finetune_strategy", "frozen"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")


def main():
    parser = argparse.ArgumentParser(description="Ablation experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size for quick smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", Path(args.config).stem)

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"{'='*60}\n")

    # Data
    data_cfg = cfg.get("data", {})
    limit = args.limit if args.limit is not None else data_cfg.get("limit", -1)
    data_loaders = get_data_loaders(
        batch_size=data_cfg.get("batch_size", 32),
        valid_size=data_cfg.get("valid_size", 0.2),
        num_workers=data_cfg.get("num_workers", 2),
        limit=limit,
        use_weighted_sampler=data_cfg.get("use_weighted_sampler", False),
        augmentation=data_cfg.get("augmentation", "full"),
    )

    # Model
    model = build_model(cfg)

    # Loss
    loss_cfg = cfg.get("loss", {})
    loss_fn = get_loss(label_smoothing=loss_cfg.get("label_smoothing", 0.0))

    # Optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = get_optimizer(
        model=model,
        optimizer=opt_cfg.get("name", "adamw"),
        learning_rate=opt_cfg.get("learning_rate", 0.001),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    # Save path
    train_cfg = cfg.get("training", {})
    save_path_template = train_cfg.get("save_path", "checkpoints/{run_name}.pt")
    save_path = save_path_template.format(run_name=run_name)
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # Google Drive backup — auto-detected when Drive is mounted on Colab
    drive_checkpoint_dir = Path("/content/drive/MyDrive/checkpoints")
    backup_path = None
    if drive_checkpoint_dir.parent.exists():
        drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        backup_path = str(drive_checkpoint_dir / Path(save_path).name)
        print(f"Google Drive backup enabled: {backup_path}")
    else:
        print("Google Drive not mounted — checkpoints saved locally only.")

    # wandb init
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb_cfg = cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "african-landmark-recognition"),
            entity=wandb_cfg.get("entity") or None,
            name=run_name,
            config=cfg,
        )

    # Train
    n_epochs = train_cfg.get("n_epochs", 30)
    optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, backup_path=backup_path)

    # Evaluate on test set
    print("\n--- Test Evaluation ---")
    test_loss, top1_acc, top5_acc, all_targets, all_preds = one_epoch_test(
        data_loaders["test"], model, loss_fn
    )

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Macro-F1: {100 * macro_f1:.1f}%")
    print("\nPer-class report:")
    class_names = data_loaders["test"].dataset.classes
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    if use_wandb:
        wandb.log({"test/macro_f1": macro_f1})

    # Export TorchScript checkpoint
    ts_path = save_path.replace(".pt", "_exported.pt")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    example = torch.randn(1, 3, 224, 224).to(device)
    try:
        traced = torch.jit.trace(model, example)
        traced.save(ts_path)
        print(f"\nTorchScript checkpoint saved to {ts_path}")
    except Exception as e:
        print(f"\nTorchScript export failed (ViT attention is not traceable): {e}")
        print("Raw state dict already saved to:", save_path)

    if use_wandb:
        wandb.finish()

    print(f"\nRun '{run_name}' complete.")
    print(f"  Top-1: {100*top1_acc:.1f}%  Top-5: {100*top5_acc:.1f}%  Macro-F1: {100*macro_f1:.1f}%")


if __name__ == "__main__":
    main()

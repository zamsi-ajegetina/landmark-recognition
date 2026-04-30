import shutil
import torch
import numpy as np
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _wandb_log(payload: dict):
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(payload)


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """Performs one training epoch. Returns average training loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    train_loss = 0.0
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader), desc="Training", total=len(train_dataloader), leave=True, ncols=80
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        train_loss += (1 / (batch_idx + 1)) * (loss_value.item() - train_loss)

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """Evaluates the model on the validation set. Returns average validation loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    valid_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader), desc="Validating", total=len(valid_dataloader), leave=True, ncols=80
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (loss_value.item() - valid_loss)

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, backup_path=None):
    """
    Full training loop with validation, early stopping, LR scheduling, and wandb logging.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    valid_loss_min = None

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | LR: {current_lr:.2e}")

        _wandb_log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": valid_loss,
            "lr": current_lr,
        })

        if valid_loss_min is None or (valid_loss_min - valid_loss) / valid_loss_min > 0.01:
            print(f"  => Validation loss improved ({valid_loss:.4f}). Saving checkpoint.")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            if backup_path:
                shutil.copy2(save_path, backup_path)
                print(f"  => Backed up to {backup_path}")

        scheduler.step()


def one_epoch_test(test_dataloader, model, loss):
    """
    Evaluates the model on the test set.

    Returns:
        test_loss (float)
        top1_acc (float): Top-1 accuracy as a fraction
        top5_acc (float): Top-5 accuracy as a fraction
        all_targets (np.ndarray): ground-truth class indices
        all_preds (np.ndarray): predicted class indices (argmax)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader), desc="Testing", total=len(test_dataloader), leave=True, ncols=80
        ):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss_value = loss(logits, target)
            test_loss += (1 / (batch_idx + 1)) * (loss_value.item() - test_loss)

            # Top-1 and Top-5
            _, top5_preds = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
            top1_preds = top5_preds[:, 0]

            top1_correct += top1_preds.eq(target).sum().item()
            top5_correct += target.unsqueeze(1).eq(top5_preds).any(dim=1).sum().item()
            total += target.size(0)

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(top1_preds.cpu().numpy())

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Top-1 Accuracy: {100 * top1_acc:.1f}% ({top1_correct}/{total})")
    print(f"Top-5 Accuracy: {100 * top5_acc:.1f}% ({top5_correct}/{total})")

    _wandb_log({
        "test/loss": test_loss,
        "test/top1_acc": top1_acc,
        "test/top5_acc": top5_acc,
    })

    return test_loss, top1_acc, top5_acc, np.array(all_targets), np.array(all_preds)

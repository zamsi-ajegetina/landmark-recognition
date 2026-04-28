import torch
import torch.nn as nn
import torch.optim


def get_loss(label_smoothing: float = 0.0):
    """
    Returns CrossEntropyLoss. Set label_smoothing > 0 for label-smoothing variant.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
):
    """
    Returns an optimizer instance.

    :param model: the model to optimize
    :param optimizer: one of 'sgd', 'adam', 'adamw'
    :param learning_rate: the learning rate
    :param momentum: the momentum (SGD) or beta1 (Adam/AdamW)
    :param weight_decay: L2 regularisation coefficient
    """
    params = model.parameters()

    if optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    elif optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=learning_rate, betas=(momentum, 0.999), weight_decay=weight_decay)

    elif optimizer.lower() == "adamw":
        # Preferred for ViT fine-tuning; decouples weight decay from gradient scaling
        return torch.optim.AdamW(params, lr=learning_rate, betas=(momentum, 0.999), weight_decay=weight_decay)

    else:
        raise ValueError(f"Optimizer '{optimizer}' not supported. Choose sgd, adam, or adamw.")

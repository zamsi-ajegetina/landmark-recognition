import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(
    model_name: str = "resnet18",
    n_classes: int = 50,
    finetune_strategy: str = "frozen",
):
    """
    Load a pretrained torchvision model and adapt its classification head.

    :param model_name: any model name from torchvision.models (e.g. resnet18, resnet50, vit_b_16)
    :param n_classes: number of output classes
    :param finetune_strategy:
        "frozen" — freeze all backbone parameters, train only the new head
        "full"   — unfreeze all parameters for end-to-end fine-tuning (use a low lr)
    :return: adapted nn.Module
    """
    if not hasattr(models, model_name):
        tv_ver = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(
            f"Model '{model_name}' not found in torchvision. "
            f"See https://pytorch.org/vision/{tv_ver}/models.html"
        )

    model_transfer = getattr(models, model_name)(weights="IMAGENET1K_V1")

    # Freeze all backbone parameters first
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Replace the classification head — handle different head conventions
    if hasattr(model_transfer, "heads"):
        # ViT family: model.heads.head is the final Linear layer
        num_ftrs = model_transfer.heads.head.in_features
        model_transfer.heads.head = nn.Linear(num_ftrs, n_classes)
    elif hasattr(model_transfer, "fc"):
        # ResNet family
        num_ftrs = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(num_ftrs, n_classes)
    elif hasattr(model_transfer, "classifier"):
        # EfficientNet / VGG family
        if isinstance(model_transfer.classifier, nn.Sequential):
            num_ftrs = model_transfer.classifier[-1].in_features
            model_transfer.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        else:
            num_ftrs = model_transfer.classifier.in_features
            model_transfer.classifier = nn.Linear(num_ftrs, n_classes)
    else:
        raise RuntimeError(f"Cannot find classification head for model '{model_name}'.")

    if finetune_strategy == "full":
        # Unfreeze all parameters for end-to-end fine-tuning
        for param in model_transfer.parameters():
            param.requires_grad = True

    return model_transfer

import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
from collections import Counter
import numpy as np

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = 1,
    limit: int = -1,
    use_weighted_sampler: bool = False,
    augmentation: str = "full",
):
    """
    Create and returns the train, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation
    :param num_workers: number of workers to use in the data loaders
    :param limit: maximum number of data points to consider (-1 = all)
    :param use_weighted_sampler: if True, use WeightedRandomSampler to handle class imbalance
    :param augmentation: "full" applies RandAugment + ColorJitter; "minimal" applies only
                         horizontal flip (used for augmentation ablation study A7)
    :return: dict with keys 'train', 'valid', 'test'
    """
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    if augmentation == "full":
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # minimal: only spatial flipping, no colour manipulation
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_transforms = {"train": train_transform, "valid": eval_transform, "test": eval_transform}

    train_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["train"])
    valid_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["valid"])

    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    if use_weighted_sampler:
        # Weight each sample inversely proportional to its class frequency
        targets = [train_data.samples[i][1] for i in train_idx.tolist()]
        class_counts = Counter(targets)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in targets]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,
    )

    test_data = datasets.ImageFolder(base_path / "test", transform=data_transforms["test"])

    if limit > 0:
        test_sampler = torch.utils.data.SubsetRandomSampler(torch.arange(min(limit, len(test_data))))
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers,
        )
    else:
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    mean, std = compute_mean_and_std()
    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
        transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
    ])
    images = inv_trans(images)
    class_names = data_loaders["train"].dataset.classes
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])

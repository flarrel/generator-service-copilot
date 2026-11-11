from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from generator_service_copilot.data import sha256_file

RANDOM_STATE = 42
IMAGE_SIZE = 128
BASELINE_SIZE = 32
PATCH_IMAGE_SIZE = 224
PATCH_GRID_SIZE = 14
RESNET18_WEIGHTS = "resnet18-f37072fd.pth"
RESNET18_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class CableDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], size: int = IMAGE_SIZE) -> None:
        self.root = root
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_image(self.root / self.paths[index], self.size)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class ResNetPatchExtractor(nn.Module):
    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.avg_pool2d(self.features(inputs), kernel_size=2)


def load_image(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(resized, dtype=np.uint8) > 0


def load_patch_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        offset = (256 - PATCH_IMAGE_SIZE) // 2
        cropped = resized.crop(
            (offset, offset, offset + PATCH_IMAGE_SIZE, offset + PATCH_IMAGE_SIZE)
        )
        array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def _set_determinism() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def train_autoencoder(
    image_root: Path,
    train_paths: list[str],
    epochs: int = 25,
    batch_size: int = 16,
) -> tuple[ConvolutionalAutoencoder, list[float]]:
    _set_determinism()
    model = ConvolutionalAutoencoder().to("cpu")
    loader = DataLoader(
        CableDataset(image_root, train_paths),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        samples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch)
            samples += len(batch)
        losses.append(running_loss / samples)
    return model.eval(), losses


def reconstruction_maps(
    model: ConvolutionalAutoencoder,
    image_root: Path,
    paths: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    loader = DataLoader(
        CableDataset(image_root, paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    maps = []
    with torch.inference_mode():
        for batch in loader:
            reconstruction = model(batch)
            maps.append(((reconstruction - batch) ** 2).mean(dim=1).numpy())
    return np.concatenate(maps)


def extract_patch_features(
    extractor: ResNetPatchExtractor,
    image_root: Path,
    paths: list[str],
    batch_size: int = 8,
) -> np.ndarray:
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack(
                [load_patch_image(image_root / path) for path in paths[start : start + batch_size]]
            )
            output = extractor(batch).permute(0, 2, 3, 1)
            output = nn.functional.normalize(output, dim=-1)
            features.append(output.numpy())
    return np.concatenate(features)


def fit_patch_memory(train_features: np.ndarray, size: int = 5_000) -> np.ndarray:
    patches = train_features.reshape(-1, train_features.shape[-1])
    generator = np.random.default_rng(RANDOM_STATE)
    selected = generator.choice(len(patches), size=min(size, len(patches)), replace=False)
    return patches[selected].astype(np.float32)


def patch_anomaly_maps(features: np.ndarray, memory: np.ndarray) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute", n_jobs=1)
    neighbors.fit(memory)
    maps = []
    for image_features in features:
        distances, _ = neighbors.kneighbors(image_features.reshape(-1, image_features.shape[-1]))
        coarse = torch.from_numpy(distances.reshape(1, 1, PATCH_GRID_SIZE, PATCH_GRID_SIZE))
        resized = nn.functional.interpolate(
            coarse,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        maps.append(resized.squeeze().numpy())
    return np.asarray(maps, dtype=np.float32)


def aggregate_scores(maps: np.ndarray, top_fraction: float) -> np.ndarray:
    flat = maps.reshape(len(maps), -1)
    count = max(1, int(flat.shape[1] * top_fraction))
    partitioned = np.partition(flat, flat.shape[1] - count, axis=1)
    return partitioned[:, -count:].mean(axis=1)


def baseline_features(image_root: Path, paths: list[str]) -> np.ndarray:
    features = []
    for relative_path in paths:
        image = load_image(image_root / relative_path, BASELINE_SIZE).numpy()
        grayscale = image.mean(axis=0)
        vertical = np.diff(grayscale, axis=0, append=grayscale[-1:])
        horizontal = np.diff(grayscale, axis=1, append=grayscale[:, -1:])
        edges = np.sqrt(vertical**2 + horizontal**2)
        features.append(np.concatenate([grayscale.ravel(), edges.ravel()]))
    return np.asarray(features, dtype=np.float32)


def fit_baseline(train_features: np.ndarray) -> Pipeline:
    components = min(64, len(train_features) - 1, train_features.shape[1])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(train_features)
    return model


def baseline_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    scaled = model.named_steps["scale"].transform(features)
    latent = model.named_steps["pca"].transform(scaled)
    reconstructed = model.named_steps["pca"].inverse_transform(latent)
    return np.mean((scaled - reconstructed) ** 2, axis=1)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == f1.max())
    chosen = best[np.argmax(recall[:-1][best])]
    return float(thresholds[chosen])


def score_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def localization_metrics(
    image_root: Path,
    development: pd.DataFrame,
    maps: np.ndarray,
    pixel_threshold: float,
) -> dict[str, float]:
    labels = []
    scores = []
    intersections = 0
    unions = 0
    for row_index, row in development.reset_index(drop=True).iterrows():
        if not row["is_anomaly"]:
            continue
        mask = load_mask(image_root / row["mask_path"], IMAGE_SIZE)
        anomaly_map = maps[row_index]
        labels.append(mask.ravel())
        scores.append(anomaly_map.ravel())
        predicted = anomaly_map >= pixel_threshold
        intersections += int(np.logical_and(predicted, mask).sum())
        unions += int(np.logical_or(predicted, mask).sum())
    all_labels = np.concatenate(labels)
    all_scores = np.concatenate(scores)
    return {
        "pixel_roc_auc_anomalous_images": float(roc_auc_score(all_labels, all_scores)),
        "pixel_average_precision_anomalous_images": float(
            average_precision_score(all_labels, all_scores)
        ),
        "micro_iou_anomalous_images": float(intersections / max(unions, 1)),
        "pixel_threshold": pixel_threshold,
    }


def select_localization_threshold(
    image_root: Path, development: pd.DataFrame, maps: np.ndarray
) -> float:
    labels = []
    scores = []
    for row_index, row in development.reset_index(drop=True).iterrows():
        if not row["is_anomaly"]:
            continue
        labels.append(load_mask(image_root / row["mask_path"], IMAGE_SIZE).ravel())
        scores.append(maps[row_index].ravel())
    return select_threshold(np.concatenate(labels), np.concatenate(scores))


def _write_plots(
    artifact_dir: Path,
    losses: list[float],
    labels: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir.parents[1] / ".mplconfig"))
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4), constrained_layout=True)
    axis = figure.subplots()
    axis.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    axis.set(title="Autoencoder training loss", xlabel="Epoch", ylabel="Mean squared error")
    figure.savefig(artifact_dir / "training_loss.png", dpi=160)

    figure = Figure(figsize=(8, 5), constrained_layout=True)
    axis = figure.subplots()
    for name, scores in candidates.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        axis.plot(recall, precision, label=name)
    axis.set(title="Development precision-recall curves", xlabel="Recall", ylabel="Precision")
    axis.legend()
    figure.savefig(artifact_dir / "development_precision_recall.png", dpi=160)


def _write_examples(
    artifact_dir: Path,
    image_root: Path,
    development: pd.DataFrame,
    maps: np.ndarray,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir.parents[1] / ".mplconfig"))
    from matplotlib.figure import Figure

    normal_index = int(np.flatnonzero(~development["is_anomaly"].to_numpy())[0])
    anomaly_index = int(np.flatnonzero(development["is_anomaly"].to_numpy())[0])
    figure = Figure(figsize=(10, 6), constrained_layout=True)
    axes = figure.subplots(2, 2)
    for row, index in enumerate((normal_index, anomaly_index)):
        image = load_image(image_root / development.iloc[index]["path"], IMAGE_SIZE)
        axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[row, 0].set_title("Normal input" if row == 0 else "Anomalous input")
        axes[row, 1].imshow(maps[index], cmap="inferno")
        axes[row, 1].set_title("Anomaly map")
        for axis in axes[row]:
            axis.axis("off")
    figure.savefig(artifact_dir / "reconstruction_examples.png", dpi=160)


def train_vision(project_root: Path, epochs: int = 25) -> dict[str, Any]:
    project_root = project_root.resolve()
    image_root = project_root / "data" / "external" / "mvtec_ad"
    manifest_path = project_root / "data" / "processed" / "mvtec_cable_manifest.csv"
    artifact_dir = project_root / "artifacts" / "vision"
    model_dir = artifact_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path).fillna("")
    train = manifest.loc[manifest["project_split"] == "train"].copy()
    development = manifest.loc[manifest["project_split"] == "development"].copy()
    if train["is_anomaly"].any() or set(train["defect_type"]) != {"good"}:
        raise ValueError("Autoencoder training data must contain only normal images")
    if (manifest["project_split"] == "test").sum() != 74:
        raise ValueError("The sealed image test split is missing")

    started = time.perf_counter()
    autoencoder, losses = train_autoencoder(image_root, train["path"].tolist(), epochs=epochs)
    train_maps = reconstruction_maps(autoencoder, image_root, train["path"].tolist())
    development_maps = reconstruction_maps(autoencoder, image_root, development["path"].tolist())
    training_seconds = time.perf_counter() - started

    labels = development["is_anomaly"].to_numpy(dtype=bool)
    candidate_scores: dict[str, np.ndarray] = {}
    candidate_maps: dict[str, np.ndarray] = {}
    candidate_reports: dict[str, Any] = {}

    train_features = baseline_features(image_root, train["path"].tolist())
    development_features = baseline_features(image_root, development["path"].tolist())
    baseline = fit_baseline(train_features)
    candidate_scores["pca_reconstruction"] = baseline_scores(baseline, development_features)

    for fraction in (1.0, 0.1, 0.05, 0.01):
        name = f"autoencoder_top_{int(fraction * 100):02d}_percent"
        candidate_scores[name] = aggregate_scores(development_maps, fraction)
        candidate_maps[name] = development_maps

    weights_path = project_root / ".torch_cache" / "hub" / "checkpoints" / RESNET18_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained weights at {weights_path}; see VISION.md for the source"
        )
    extractor = ResNetPatchExtractor(weights_path).eval()
    train_patch_features = extract_patch_features(extractor, image_root, train["path"].tolist())
    development_patch_features = extract_patch_features(
        extractor, image_root, development["path"].tolist()
    )
    patch_memory = fit_patch_memory(train_patch_features)
    patch_maps = patch_anomaly_maps(development_patch_features, patch_memory)
    for fraction in (0.1, 0.05, 0.01):
        name = f"resnet_patch_top_{int(fraction * 100):02d}_percent"
        candidate_scores[name] = aggregate_scores(patch_maps, fraction)
        candidate_maps[name] = patch_maps

    for name, scores in candidate_scores.items():
        threshold = select_threshold(labels, scores)
        candidate_reports[name] = score_metrics(labels, scores, threshold)

    selected = max(
        candidate_reports,
        key=lambda name: (
            candidate_reports[name]["roc_auc"],
            candidate_reports[name]["average_precision"],
        ),
    )
    localization_method = selected
    localization_maps = candidate_maps.get(selected)
    if localization_maps is None:
        localization_method = "autoencoder_top_01_percent"
        localization_maps = development_maps
    pixel_threshold = select_localization_threshold(image_root, development, localization_maps)
    localization = localization_metrics(image_root, development, localization_maps, pixel_threshold)
    localization["method"] = localization_method

    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "image_size": IMAGE_SIZE,
            "architecture": "ConvolutionalAutoencoder",
        },
        model_dir / "autoencoder.pt",
    )
    joblib.dump(baseline, model_dir / "pca_reconstruction.joblib")
    np.save(model_dir / "train_map_reference.npy", train_maps.mean(axis=0))
    np.save(model_dir / "patch_memory.npy", patch_memory)

    predictions = development[["path", "defect_type", "is_anomaly"]].copy()
    for name, scores in candidate_scores.items():
        predictions[f"score_{name}"] = scores
    predictions.to_csv(artifact_dir / "development_predictions.csv", index=False)

    report: dict[str, Any] = {
        "sealed_test_used": False,
        "device": "cpu",
        "data": {
            "training_normal_images": len(train),
            "development_normal_images": int((~labels).sum()),
            "development_anomaly_images": int(labels.sum()),
            "test_images_evaluated": 0,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "autoencoder": {
            "architecture": "3-layer convolutional encoder and transposed-convolution decoder",
            "parameters": sum(parameter.numel() for parameter in autoencoder.parameters()),
            "epochs": epochs,
            "batch_size": 16,
            "optimizer": "Adam(lr=0.001, weight_decay=0.00001)",
            "final_training_loss": losses[-1],
            "training_seconds": training_seconds,
        },
        "pretrained_patch_features": {
            "backbone": "torchvision ResNet-18 ImageNet-1K v1",
            "weights_url": RESNET18_WEIGHTS_URL,
            "weights_sha256": sha256_file(weights_path),
            "weights_committed": False,
            "feature_layer": "layer2 pooled to 14x14",
            "feature_dimensions": int(patch_memory.shape[1]),
            "memory_patches": int(len(patch_memory)),
        },
        "candidates": candidate_reports,
        "selected_candidate": selected,
        "selected_metrics": candidate_reports[selected],
        "localization": localization,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_plots(artifact_dir, losses, labels, candidate_scores)
    _write_examples(artifact_dir, image_root, development, localization_maps)

    artifact_paths = [
        artifact_dir / "report.json",
        artifact_dir / "development_predictions.csv",
        artifact_dir / "training_loss.png",
        artifact_dir / "development_precision_recall.png",
        artifact_dir / "reconstruction_examples.png",
        model_dir / "autoencoder.pt",
        model_dir / "pca_reconstruction.joblib",
        model_dir / "train_map_reference.npy",
        model_dir / "patch_memory.npy",
    ]
    contract = {
        "schema_version": 1,
        "sealed_test_used": False,
        "selected_candidate": selected,
        "selected_threshold": candidate_reports[selected]["threshold"],
        "image_size": IMAGE_SIZE,
        "files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    (artifact_dir / "artifact_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_autoencoder(path: Path) -> ConvolutionalAutoencoder:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = ConvolutionalAutoencoder()
    model.load_state_dict(artifact["state_dict"])
    return model.eval()


def predict_autoencoder(path: Path, model_path: Path, top_fraction: float) -> dict[str, Any]:
    model = load_autoencoder(model_path)
    image = load_image(path, IMAGE_SIZE).unsqueeze(0)
    with torch.inference_mode():
        reconstruction = model(image)
        anomaly_map = ((reconstruction - image) ** 2).mean(dim=1).numpy()
    return {
        "score": float(aggregate_scores(anomaly_map, top_fraction)[0]),
        "anomaly_map": anomaly_map[0],
    }


def predict_patch(
    path: Path,
    weights_path: Path,
    memory_path: Path,
    top_fraction: float = 0.1,
) -> dict[str, Any]:
    extractor = ResNetPatchExtractor(weights_path).eval()
    image_root = path.parent
    features = extract_patch_features(extractor, image_root, [path.name])
    anomaly_map = patch_anomaly_maps(features, np.load(memory_path))
    return {
        "score": float(aggregate_scores(anomaly_map, top_fraction)[0]),
        "anomaly_map": anomaly_map[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate visual anomaly models")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(train_vision(args.project_root, epochs=args.epochs), indent=2))


if __name__ == "__main__":
    main()

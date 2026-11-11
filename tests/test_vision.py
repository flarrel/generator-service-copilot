from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generator_service_copilot.vision import (
    ConvolutionalAutoencoder,
    aggregate_scores,
    fit_patch_memory,
    patch_anomaly_maps,
    select_threshold,
)


def test_autoencoder_preserves_image_shape() -> None:
    model = ConvolutionalAutoencoder()
    inputs = torch.rand(2, 3, 128, 128)

    outputs = model(inputs)
    assert outputs.shape == inputs.shape
    assert outputs.min() >= 0
    assert outputs.max() <= 1


def test_aggregate_scores_focuses_on_largest_errors() -> None:
    maps = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)

    assert aggregate_scores(maps, 1.0)[0] == 1.5
    assert aggregate_scores(maps, 0.25)[0] == 3.0


def test_threshold_selection_separates_simple_scores() -> None:
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(labels, scores)

    assert 0.2 < threshold <= 0.8


def test_no_test_image_is_loaded_by_training_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(root / "data" / "processed" / "mvtec_cable_manifest.csv")

    assert (manifest["project_split"] == "test").sum() == 74


def test_patch_memory_sampling_is_bounded_and_deterministic() -> None:
    features = np.arange(2 * 14 * 14 * 4, dtype=np.float32).reshape(2, 14, 14, 4)

    first = fit_patch_memory(features, size=100)
    second = fit_patch_memory(features, size=100)

    assert first.shape == (100, 4)
    np.testing.assert_array_equal(first, second)


def test_patch_anomaly_maps_are_resized_for_inference() -> None:
    features = np.zeros((2, 14, 14, 3), dtype=np.float32)
    memory = np.ones((5, 3), dtype=np.float32)

    maps = patch_anomaly_maps(features, memory)

    assert maps.shape == (2, 128, 128)
    assert np.all(maps > 0)

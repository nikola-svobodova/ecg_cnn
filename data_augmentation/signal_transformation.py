"""Signal transformations

This file is used to implement several signal transformations used in data augmentation.
"""

import numpy as np
from utils.constants import NUM_OF_SAMPLES


def add_baseline_wander(original, amplitude=0.05, fun_endpoint=20, phase_shift=0):
    cos_endpoint = fun_endpoint * np.pi
    baseline_drift = amplitude * np.cos(np.linspace(0, cos_endpoint, NUM_OF_SAMPLES) - phase_shift * np.pi)
    transformed = original + baseline_drift
    return transformed


def add_gaussian_noise(original, mu=0, sigma=0.05):
    noise = np.random.normal(mu, sigma, NUM_OF_SAMPLES)
    transformed = original + noise
    return transformed


def scale(original, scaling_factor=1.2):
    transformed = original * scaling_factor
    return transformed


def dropout(original, rate=0.05):
    transformed = original.copy()
    dropout_indices = np.random.choice(np.arange(NUM_OF_SAMPLES), size=int(NUM_OF_SAMPLES * rate))
    transformed[dropout_indices] = 0
    return transformed


def horizontal_flip(original):
    transformed = np.flip(original)
    return transformed


def vertical_flip(original):
    transformed = -original
    return transformed


def permutation(original, num_subsections=20):
    subsections = np.array_split(original, num_subsections)
    np.random.shuffle(subsections)
    transformed = np.concatenate(subsections)
    return transformed

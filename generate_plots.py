"""generate_plots module

Contains utility to generate synthetic sensor data matching the notebook's
parameters.
"""

import numpy as np


def generate_data(seed):
    """Generate synthetic temperature sensor data.

    Parameters
    ----------
    seed : int
        Random seed for reproducible noise and dropouts.

    Returns
    -------
    sensor_a : ndarray, shape (n_samples,)
        Temperature readings from sensor A (float64). May contain NaNs for simulated dropouts.
    sensor_b : ndarray, shape (n_samples,)
        Temperature readings from sensor B (float64). May contain NaNs for simulated dropouts.
    timestamps : ndarray, shape (n_samples,)
        Time stamps in seconds (float64) from 0 to duration-1.

    Notes
    -----
    This function reproduces the notebook behavior: 1 Hz sampling for one hour
    (3600 samples), a 30-minute sinusoidal baseline, additive Gaussian noise,
    a small bias on sensor_b, random sparse dropouts (~2%), and one contiguous
    gap of 60 samples. Returned arrays use dtype float64.
    """
    rng = np.random.default_rng(seed)
    fs = 1  # Hz
    duration_s = 3600  # one hour
    t = np.arange(0, duration_s, 1/fs, dtype=float)
    temp_base = 22.0
    # 30-minute sinusoid
    temp = temp_base + 0.6 * np.sin(2 * np.pi * t / 1800.0)

    noise_std = 0.5
    bias = 0.2

    # sensor signals
    sensor_a = temp + rng.normal(0.0, noise_std, size=t.shape)
    sensor_b = temp + rng.normal(0.0, noise_std, size=t.shape) + bias

    # random sparse dropouts (~2%)
    drop_prob = 0.02
    drop_mask = rng.random(t.shape) < float(drop_prob)
    sensor_a[drop_mask] = np.nan
    sensor_b[drop_mask] = np.nan

    # one contiguous gap of 60 samples
    gap_length = 60
    if gap_length > 0 and len(t) > gap_length:
        start = int(rng.integers(0, max(1, len(t) - gap_length)))
        sensor_a[start:start+gap_length] = np.nan
        sensor_b[start:start+gap_length] = np.nan

    return sensor_a.astype(np.float64), sensor_b.astype(np.float64), t.astype(np.float64)


if __name__ == '__main__':
    a, b, ts = generate_data(0)
    print('Generated', a.shape, b.shape, ts.shape)

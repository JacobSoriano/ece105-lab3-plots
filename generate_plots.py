"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data using NumPy
and produces scatter, histogram, and box plot visualizations saved
as PNG files.

Usage
-----
    python generate_plots.py
"""
# Create a function generate_data(seed) that returns sensor_a, sensor_b,
# and timestamps arrays with the same parameters as in the notebook.
# Use NumPy-style docstring with Parameters and Returns sections.

from matplotlib.pylab import seed


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
    import numpy as np
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

# Create plot_scatter(sensor_a, sensor_b, timestamps, ax) that draws
# the scatter plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.
def plot_scatter(ax, sensor_a, sensor_b, label_a='Sensor A', label_b='Sensor B', scatter_kwargs=None, identity=True):  
    """Plot Sensor A vs Sensor B on an existing Axes
    in-place.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        An existing Matplotlib Axes object to draw on (modified in-place).
    sensor_a : ndarray
        1-D array of readings from sensor A (float-like). May contain NaNs.
    sensor_b : ndarray
        1-D array of readings from sensor B (float-like). May contain NaNs.
    label_a : str, optional
        Label for sensor A (used in legend). Default 'Sensor A'.
    label_b : str, optional
        Label for sensor B (used in legend). Default 'Sensor B'.
    scatter_kwargs : dict or None, optional
        Additional keyword arguments forwarded to Axes.scatter. Default None.
    identity : bool, optional
        If True, draw a dashed identity line (y=x) to aid comparison. Default True.
   
    Returns
    -------
    None
    The function modifies the provided Axes in-place and returns None.
    """
    import numpy as _np
   
    # Prepare plotting kwargs
    kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)
    # Mask out samples where either sensor is NaN
    mask = _np.isfinite(sensor_a) & _np.isfinite(sensor_b)
    xa = _np.asarray(sensor_a)[mask]
    xb = _np.asarray(sensor_b)[mask]
   
    # Scatter
    ax.scatter(xa, xb, alpha=0.6, s=18, label=f"{label_a} vs {label_b}", **kwargs)
   
    # Identity line
    if identity and xa.size > 0 and xb.size > 0:
        vmin = min(xa.min(), xb.min())
        vmax = max(xa.max(), xb.max())
        ax.plot([vmin, vmax], [vmin, vmax], color='gray', linestyle='--', linewidth=1)
   
    ax.set_xlabel(f"{label_a} (°C)")
    ax.set_ylabel(f"{label_b} (°C)")
    ax.set_title(f"{label_a} vs {label_b} (scatter)")
    ax.grid(alpha=0.3)
    # do not return Axes (in-place)
    return None

# Create plot_histogram(sensor_a, sensor_b, timestamps, ax) that draws
# the histogram plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

def plot_histogram(ax, sensor_a, sensor_b, bins=30, alpha=0.6, label_a='Sensor A', label_b='Sensor B', hist_kwargs=None, show_means=True):
    """Plot overlaid histograms of sensor_a and sensor_b on an Axes in-place.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib Axes to draw on (modified in-place).
    sensor_a : ndarray
        1-D array of readings from sensor A (may contain NaNs).
    sensor_b : ndarray
        1-D array of readings from sensor B (may contain NaNs).
    bins : int or sequence, optional
        Number of histogram bins or bin edges. Default is 30.
    alpha : float, optional
        Transparency for the histogram bars. Default 0.6.
    label_a, label_b : str, optional
        Labels for the two sensors used in the legend.
    hist_kwargs : dict or None, optional
        Additional keyword arguments forwarded to Axes.hist for both plots.
    show_means : bool, optional
        If True, draw dashed vertical lines at each sensor's mean.

    Returns
    -------
    None
        The function modifies the provided Axes in-place and returns None.
    """
    import numpy as _np

    kwargs = {} if hist_kwargs is None else dict(hist_kwargs)

    # Mask NaNs
    a = _np.asarray(sensor_a)
    b = _np.asarray(sensor_b)
    a_clean = a[_np.isfinite(a)]
    b_clean = b[_np.isfinite(b)]

    # Plot histograms (overlaid)
    ax.hist(a_clean, bins=bins, alpha=alpha, label=label_a, **kwargs)
    ax.hist(b_clean, bins=bins, alpha=alpha, label=label_b, **kwargs)

    # Mean lines
    if show_means and a_clean.size > 0:
        ax.axvline(a_clean.mean(), color='tab:blue', linestyle='--', linewidth=1)
    if show_means and b_clean.size > 0:
        ax.axvline(b_clean.mean(), color='tab:orange', linestyle='--', linewidth=1)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram: {label_a} and {label_b}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return None

# Create plot_boxplot(sensor_a, sensor_b, timestamps, ax) that draws
# the box plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

def plot_boxplot(ax, sensor_a, sensor_b, labels=('Sensor A','Sensor B'), box_kwargs=None, show_points=True):
    """Plot box plots for two sensors on an Axes,
    in-place.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib Axes to draw on (modified in-place).
    sensor_a : ndarray
        1-D array of readings from sensor A (may contain NaNs).
    sensor_b : ndarray
        1-D array of readings from sensor B (may contain NaNs).
    labels : sequence of str, optional
        Labels for the two boxes; default ("Sensor A","Sensor B").
    box_kwargs : dict or None, optional
        Additional keyword arguments forwarded to Axes.boxplot.
    show_points : bool, optional
        If True, draw jittered individual data points over the boxes.
   
    Returns
    -------
    None
        Modifies ax in-place and returns None.
    """
    import numpy as _np
   
    kwargs = {} if box_kwargs is None else dict(box_kwargs)
   
    a = _np.asarray(sensor_a)
    b = _np.asarray(sensor_b)
    a_clean = a[_np.isfinite(a)]
    b_clean = b[_np.isfinite(b)]
   
    data = [a_clean, b_clean]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
    boxprops=dict(facecolor='lightgray', color='k'), **kwargs)
   
    if show_points:
        # jitter points for visibility
        xa = _np.random.normal(1, 0.06, size=a_clean.size)
        xb = _np.random.normal(2, 0.06, size=b_clean.size)
        ax.scatter(xa, a_clean, alpha=0.6, color='tab:blue', s=10)
        ax.scatter(xb, b_clean, alpha=0.6, color='tab:orange', s=10)
   
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Box plot of {labels[0]} and {labels[1]}')
    ax.grid(axis='y', alpha=0.3)
    return None

# Create main() that generates data, creates a 1x3 subplot figure,
# calls each plot function, adjusts layout, and saves as sensor_analysis.png
# at 150 DPI with tight bounding box.
def main(seed=0, out_dir=None):
    """Generate data and create/save scatter, histogram, and boxplot figures.

    Parameters
    ----------
    seed : int, optional
        Random seed used to generate synthetic data. Default is 0.
    out_dir : str or pathlib.Path, optional
        Directory to write output PNG files. If None, uses the current
        working directory.

    Returns
    -------
    None
        Saves three PNG files: sensor_scatter.png, sensor_hist.png, sensor_box.png
        in the output directory.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir) if out_dir is not None else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    sensor_a, sensor_b, timestamps = generate_data(seed)

    # Scatter
    fig, ax = plt.subplots(figsize=(6,5))
    plot_scatter(ax, sensor_a, sensor_b)
    scatter_path = out_dir / 'sensor_scatter.png'
    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(7,4))
    plot_histogram(ax, sensor_a, sensor_b)
    hist_path = out_dir / 'sensor_hist.png'
    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(7,5))
    plot_boxplot(ax, sensor_a, sensor_b)
    box_path = out_dir / 'sensor_box.png'
    fig.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {scatter_path}, {hist_path}, {box_path}")

    # composite 1x3 figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_scatter(axs[0], sensor_a, sensor_b)
    plot_histogram(axs[1], sensor_a, sensor_b)
    plot_boxplot(axs[2], sensor_a, sensor_b)
    for a in axs:
        a.label_outer()
    composite_path = out_dir / 'sensor_analysis.png'
    fig.savefig(composite_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved composite: {composite_path}")


if __name__ == '__main__':
    main()

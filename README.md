# Sensor Plots — Synthetic Data Analysis

A small utility that generates synthetic temperature sensor data and produces publication-quality plots (scatter, histogram, and boxplot) for quick analysis and demonstration.

## Installation

1. Activate the project environment (recommended):

   conda activate ece105

2. Install required packages with conda or mamba:

   conda install -c conda-forge numpy matplotlib

   or

   mamba install -c conda-forge numpy matplotlib

(If you don't use conda, you can use pip in a virtual environment: pip install numpy matplotlib.)

## Usage

Run the script from the project folder:

python generate_plots.py

By default this uses a reproducible random seed and writes output PNG files to the current directory. You can call main(seed=<int>, out_dir='<path>') from Python to customize the seed and output directory.

## Example output

The script produces three individual plot files plus a composite:

- `sensor_scatter.png` — scatter plot of Sensor A vs Sensor B with an identity guideline (y=x).
- `sensor_hist.png` — overlaid histograms of Sensor A and Sensor B with mean lines.
- `sensor_box.png` — boxplots for each sensor with jittered individual points.
- `sensor_analysis.png` — a 1×3 composite figure combining the scatter, histogram, and boxplot.

These files are saved at 150 DPI with tight bounding boxes for inclusion in reports.

## AI tools used and disclosure

(Placeholder — describe any AI tools, code assistants, or automated edits used here.)

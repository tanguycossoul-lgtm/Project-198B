# Leaflet Angle Pipeline

Measures heart valve leaflet angle frame-by-frame from a PNG image sequence.
Each frame is dispatched to one of two detection methods based on leaflet visibility,
and results are written to a smoothed CSV.

---

## Project Structure

```
project/
├── main.py                     # Entry point — run this to process a dataset
├── config_default.json         # Default calibration parameters (all datasets)
│
├── pipeline/                   # Core processing library (imported by main and tools)
│   ├── laser_detection.py      # Method 1: laser Gaussian peak detection
│   ├── leaflet_detection.py    # Method 2: leaflet segmentation + circle fit
│   ├── config_loader.py        # JSON config load / save
│   └── smooth_results.py       # Butterworth low-pass angle smoothing
│
├── tools/                      # Standalone utility scripts
│   ├── config_ui.py            # PySide6 GUI for tuning and saving config
│   └── convert_png_to_video.py # Convert a PNG sequence to an MP4 video
│
├── data/                       # Input data — one subfolder per dataset
│   └── <dataset>/
│       ├── Set_01_XXXX.png     # Input frames (numbered)
│       └── config_custom.json  # Optional dataset-specific config override
│
├── output/                     # Generated outputs (recreated each run)
│   └── results.csv
└── dbg/                        # Debug images (recreated each run)
```

---

## Detection Overview

For every frame, the pipeline checks the white pixel count (WPC) inside the
leaflet crop region to decide which method to apply:

| WPC vs threshold | Method used | When |
|---|---|---|
| `WPC < method_select_thresh` | **Method 1 — laser** | Leaflet not visible; laser reflection still detectable |
| `WPC ≥ method_select_thresh` | **Method 2 — leaflet** | Leaflet visible and segmentable |

**Method 1 — laser (`pipeline/laser_detection.py`)**
Sweeps the laser crop region in vertical strips, fits a Gaussian to each strip's
intensity profile to locate the reflected laser beam at sub-pixel precision, then
fits a line through the peaks. Leaflet angle is derived from mirror reflection
geometry: `angle = ((reflected_angle + primary_laser_angle) / 2) − 90`.

**Method 2 — leaflet (`pipeline/leaflet_detection.py`)**
Thresholds the leaflet crop to a binary mask, records the centroid and leftmost
white pixel for each frame. After all frames are processed, a circle is fitted to
the leftmost-pixel trajectory; individual frame angles are then computed as the
arctangent of each point relative to the fitted circle centre.

---

## Dependencies

```
pip install -r requirements.txt
```

Core: `opencv-python`, `numpy`, `scipy`
UI tool only: `PySide6`

---

## Usage

### `main.py` — run the pipeline

```bash
python main.py -d <DATASET>
```

`<DATASET>` is the name of a subfolder inside `data/`.

```bash
# Example
python main.py -d "Leaflet angle-10-21-2025-Set_01"
```

**What it does:**
1. Validates that `data/<DATASET>/` exists and contains PNG files.
2. Loads `data/<DATASET>/config_custom.json` if present, otherwise falls back to `config_default.json`.
3. Detects the primary (incident) laser angle from the first frame.
4. Processes each frame in `frame_range`, dispatching to Method 1 or Method 2.
5. Fits a circle to the leaflet trajectory and resolves all Method 2 angles.
6. Applies a Butterworth low-pass filter to smooth the angle series.
7. Writes `output/results.csv`.

**Output CSV columns:**

| Column | Description |
|---|---|
| `Frame` | Frame number |
| `Angle` | Raw angle (degrees) |
| `Angle (<cutoff>)` | Smoothed angle at the configured cutoff |
| `Method` | `laser` or `leaflet` |
| `Angle (laser)` | Raw angle if method is laser, else blank |
| `Angle (leaflet)` | Raw angle if method is leaflet, else blank |

---

## Configuration

All parameters are stored in JSON files with `{"value": ..., "description": ...}` entries.

### Config resolution order

1. `data/<DATASET>/config_custom.json` — dataset-specific override (created by the UI)
2. `config_default.json` — project-wide defaults (fallback)

### Parameters

| Key | Default | Description |
|---|---|---|
| `frame_range` | `[0, 2000]` | Inclusive range of frame numbers to process |
| `method_select_thresh` | `400` | WPC threshold: below → laser method, at/above → leaflet method |
| `smooth_cutoff` | `0.1` | Low-pass filter cutoff as fraction of Nyquist (0.0–1.0) |
| `img_laser_prim_tl/br` | `[1200,5]` / `[1800,1000]` | Crop region for primary laser angle detection |
| `img_laser_tl/br` | `[500,5]` / `[1200,1650]` | Crop region for Method 1 (reflected laser) |
| `img_laser_thresh` | `90` | Intensity threshold for laser detection |
| `laser_strip_width` | `10` | Vertical strip width for Gaussian peak detection (px) |
| `laser_peak_step` | `10` | Horizontal step between strips (px) |
| `laser_min_peak` | `100` | Min strip peak intensity to attempt a Gaussian fit |
| `img_leafl_tl/br` | `[1450,660]` / `[1615,1240]` | Crop region for Method 2 (leaflet segmentation) |
| `img_leafl_thresh` | `20` | Binary threshold for leaflet segmentation |
| `leaflet_calib_frame_range` | `[0, 200]` | Frame range used to fit the circle trajectory |

---

## Tools

### `tools/config_ui.py` — interactive config tuning

PySide6 GUI for visually setting crop regions, thresholds, and ranges,
then saving a `config_custom.json` into the selected dataset folder.

```bash
python tools/config_ui.py [--data data]
```

**Tabs:**
- **Crops & Thresholds** — click on the image to set crop rectangle corners (TL then BR)
  for each region; adjust laser/leaflet/method-select thresholds with sliders.
  Live white-pixel count shows which method would be triggered on the current frame.
  Mask overlay visualises the thresholded region. Manual 3-point circle fit tool for
  verifying the leaflet trajectory.
- **Ranges & Other** — set frame range, calibration frame range, strip parameters,
  and smooth cutoff.

**Buttons:**
- **Restore default config** — reloads `config_default.json` into the UI (does not save).
- **Save custom config** — writes `data/<dataset>/config_custom.json`.

### `tools/convert_png_to_video.py` — PNG sequence to MP4

Converts a numbered PNG sequence into a browser-compatible MP4.

```bash
python tools/convert_png_to_video.py
```

Configure `frame_range`, `repeat_count`, `repeat_delay`, and `output_path` at
the bottom of the script. Optionally stamps frame numbers on each frame.

---

## Pipeline Modules (`pipeline/`)

| Module | Purpose |
|---|---|
| `laser_detection.py` | CLAHE enhancement, Gaussian peak detection, line fitting, primary and reflected angle calculation |
| `leaflet_detection.py` | Binary thresholding, centroid/leftmost pixel detection, algebraic circle fit, angle computation |
| `config_loader.py` | `load_config(path)` → `{key: value}` dict; `save_config(cfg, path)` preserving descriptions |
| `smooth_results.py` | `smooth_results(results, cutoff)` — zero-phase Butterworth filter; also runnable standalone for cutoff sweeps |

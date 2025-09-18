# NeRF / 3D Capture Preprocessing GUI

Modern PyQt6 application to batch preprocess multi-view capture folders for NeRF / Gaussian Splatting style pipelines. Now supports fully flexible, recursive image discovery with custom filename glob patterns (no strict naming required).

It performs:

- AI foreground segmentation (rembg) or classical thresholding
- Tight bounding box cropping (with absolute + relative padding, square option, union mode)
- Optional RGBA with transparency OR opaque alpha OR simple bbox-only RGB crops
- Mask export, debug visual panels (original | overlay | crop)
- Bayer demosaic (raw grayscale to RGB) if needed
- Preset-driven one-click workflows

## Folder Layout (Legacy Example)

```
<root_input>/
  object_or_session_A/
    capture_001/
      cam_01_XXXXXXXX.tiff
      cam_02_XXXXXXXX.tiff
      ...
    capture_002/
      ...
  object_or_session_B/
    capture_001/ ...
```

This structured layout is still supported, but no longer required. You may point the input root at any directory containing one or more scene subfolders (or treat each subfolder itself as a scene). Recursive search + filename patterns allow arbitrary naming schemes (see Flexible Image Discovery). If you point at a folder whose immediate children are `capture_###`, each capture becomes a scene automatically.

## Quick Start (Source)

```powershell
# (Recommended) create environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python gui_app.py
```

First AI segmentation run downloads the rembg model (one time).

## Core Concepts

- **Scene**: Either an object/session directory containing `capture_*` subfolders OR a single `capture_*` folder (each capture becomes a scene if you set root above them).
- **Capture images**: Filenames matched by glob pattern (default `cam_*.tiff`).
- **Mask**: Either AI (preferred) or threshold-based (Otsu / percentile) refined with morphology & largest component.
- **Bounding Box**: Derived from mask; can be padded (pixels + relative %) and squared; can optionally be a union across images.

## Presets

| Preset                      | Purpose                                    | Key Settings                                |
| --------------------------- | ------------------------------------------ | ------------------------------------------- |
| AI Tight RGBA (transparent) | High-quality transparent foreground assets | AI seg on, RGBA+mask, small pad (4px + 10%) |
| AI Tight Crop (RGB only)    | Just cropped RGB (no alpha)                | AI seg on, bbox-only, small pad             |
| AI Opaque RGBA              | RGBA with full alpha background retained   | AI seg on, RGBA opaque, moderate pad        |
| Classical Threshold Crop    | No AI dependency                           | AI off (Otsu), RGBA+mask                    |
| BBox Only Fast              | Speedy dataset cropping                    | AI seg on, bbox-only, moderate pad          |
| Custom                      | Any manual adjustments after a preset      | —                                          |

You can tweak after choosing a preset. (Currently label stays; future change could auto-switch to "Custom".)

## Key Options

- **AI Segmentation**: Use rembg for robust foreground detection.
- **Padding (px)**: Absolute pixels added each side.
- **Extra Space (%)**: Relative expansion per side (percentage of bbox size).
- **Consistent Crop (Union)**: Single union bbox for all images in a scene.
- **Force Square**: Expand bbox to square (fills with existing background).
- **Save Mask**: Persist `*_mask.png` (cropped).
- **Save RGBA (Alpha)**: Output `*_rgba.png`; transparent unless RGBA Opaque is enabled.
- **RGBA Opaque**: Alpha channel forced to 255 (keep background, still 4 channels).
- **Save Cropped RGB**: Plain cropped RGB (`*_crop.png`).
- **BBox Only (simple crop)**: Shortcut: only saves cropped RGB (suppresses mask/rgba) but still relies on AI/classical mask for bbox.
- **Demosaic (Bayer)**: Choose pattern if source TIFFs are single-channel raw mosaic.
- **Limit Images**: Process only first N per scene (0 = all).
- **Debug Panels**: Side-by-side visualization: original | mask overlay | crop.
- **Recursive Search**: (GUI) When enabled, traverse all subdirectories under each scene to find images.
- **Filename Patterns**: (GUI) Semicolon-separated glob patterns (e.g. `*.png;*.jpg;cam_*.tif`). Leave empty to accept ANY supported image extension: `.png .jpg .jpeg .tif .tiff .bmp`.

## Flexible Image Discovery (Recursive + Patterns)

The GUI provides two fields enabling naming‑agnostic datasets:

- Recursive Search (checkbox)
- Filename Patterns (line edit; semicolon separated)

Behavior Summary:

1. If patterns field is NON-empty: each segment is applied as a glob (case-sensitive patterns, case-insensitive extension filtering). Only matches from those patterns are considered.
2. If patterns field is EMPTY: all files with supported extensions inside the scene (and optionally its subfolders if recursive) are collected.
3. Non-recursive + empty patterns maintains backward compatibility by still looking into `capture_*` subfolders first (any images), then images directly under the scene directory.
4. Results are deduplicated and sorted (stable ordering for reproducible processing).

Examples:

| Goal                                                  | Patterns                         | Recursive |
| ----------------------------------------------------- | -------------------------------- | --------- |
| Any images in flat capture folders                    | (leave empty)                    | Off       |
| Only TIFF camera frames                               | `cam_*.tiff`                   | Off       |
| Mixed PNG/JPEG arbitrary names                        | `*.png;*.jpg`                  | Off or On |
| Deep directory of frames with multiple naming schemes | `*.png;cam_*.tif;frame_??.jpg` | On        |
| Accept everything anywhere (supported types)          | (empty)                          | On        |

Tips:

- Keep output (`processed_YYYY...`) outside of input root to avoid re-discovery. If you must place outputs inside, use a different parent directory or later we can add exclude patterns (open issue welcome).
- Patterns are applied relative to each scene root. With recursion enabled, globs match against full relative paths via `Path.rglob` semantics.
- Unsupported formats (e.g. EXR) are currently ignored; open a PR to extend the extension set.

Current CLI note: recursive + multi-pattern discovery is exposed in the GUI. (CLI flags for these can be added; open an issue if you need them.)

## Workflow Examples

### 1. Transparent Assets for Compositing

Preset: "AI Tight RGBA (transparent)" -> Select scenes -> Run.
Outputs: `*_rgba.png` with alpha, `*_mask.png`, `*_debug.png`.

### 2. Fast Dataset Cropping (No Alpha)

Preset: "BBox Only Fast" -> Run.
Outputs: `*_crop.png` (+ debug if enabled).

### 3. Keep Full Background but Crop Tightly

Preset: "AI Opaque RGBA" -> Run.
Outputs: `*_rgba.png` (alpha=255), `*_debug.png`.

### 4. Classical (Offline / No Model Download)

Disable AI Segmentation or use preset "Classical Threshold Crop".

## Output Structure

```
<output_root>/
  <scene_name>/
    metadata.json
    cam_01_xxxx_crop.png       (if enabled / bbox-only)
    cam_01_xxxx_rgba.png       (if enabled)
    cam_01_xxxx_mask.png       (if enabled)
    cam_01_xxxx_debug.png      (if debug)
```

`metadata.json` contains per-image bbox + crop dimensions and union bbox (if used).

## Notes on AI Segmentation

- Uses rembg (U2Net default). First run downloads model (~50–100MB).
- For offline distribution you may pre-run once, then copy the model cache (`%USERPROFILE%\.u2net`).

## Performance Tips

- Enable **BBox Only** for fastest large batch (skips PNG with alpha & mask writes).
- Reduce **Extra Space (%)** for tighter crops (training sometimes benefits from small context, e.g. 10–15%).
- Union BBox is helpful for consistent framing in multi-view reconstruction stability.

## RGBA Modes Summary

| Mode        | Save RGBA | RGBA Opaque | BBox Only | Result                                   |
| ----------- | --------- | ----------- | --------- | ---------------------------------------- |
| Transparent | Yes       | No          | No        | Foreground with alpha around object      |
| Opaque      | Yes       | Yes         | No        | Full image area in crop with solid alpha |
| BBox Only   | (Ignored) | (Ignored)   | Yes       | Only `*_crop.png`                      |

## Building a Windows Executable (Optional)

### Direct PyInstaller Invocation
```powershell
pip install pyinstaller
pyinstaller gui_app.py --name nerf-preprocess-gui --noconfirm --clean --add-data preprocess.py;.
```
Artifacts:
- Folder build: `dist/nerf-preprocess-gui/` (recommended for easier debugging & smaller delta updates)
- Add `--onefile` for a single EXE (slower first launch; extracts to temp)

### Automated Script (`build_exe.ps1`)
Convenience build & zip packaging script is included.
```powershell
# Folder distribution (faster startup, inspectable files)
./build_exe.ps1 -Name nerf-preprocess-gui

# One-file executable
./build_exe.ps1 -Name nerf-preprocess-gui -OneFile

# Rebuild virtual environment from scratch
./build_exe.ps1 -RebuildVenv

# Skip dependency reinstall (faster incremental)
./build_exe.ps1 -SkipInstall
```
Outputs (auto-zipped):
- `nerf-preprocess-gui_folder_YYYYMMDD_HHMMSS.zip`
- `nerf-preprocess-gui_onefile_YYYYMMDD_HHMMSS.zip`

Script steps:
1. Create / reuse `.venv`
2. (Re)install dependencies unless `-SkipInstall`
3. Run PyInstaller (`preprocess.py` embedded as data)
4. Compress distribution to a timestamped zip

Model Cache Note: rembg model weights are cached in `%USERPROFILE%\.u2net` and not bundled. Run once online or pre-seed that folder for offline machines.

### .gitignore
The project includes a `.gitignore` excluding build artifacts (`dist/`, `build/`, `*.spec`), virtual environments (`.venv/`), processed output (`processed_*/`), and caches. Adjust if you add new tooling or artifact directories.

## GitHub Setup (Example)

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Tag & release:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

Attach zipped `dist/nerf-preprocess` or the one-file exe to the GitHub Release.

## Repository Naming Guidance

Choose a concise, searchable name reflecting NeRF / multi-view preprocessing, foreground cropping, and segmentation. Avoid overly generic terms like "tools" or "scripts" alone. Recommended patterns: `<purpose>-<domain>` or `<domain>-<action>`.

Criteria:

- Descriptive (signals NeRF / multi-view / cropping)
- Short (ideally <= 20 chars excluding hyphens)
- Lowercase with hyphens
- Avoid trademarked names or ambiguous abbreviations

Suggested Names (pick one or adapt):

- `nerf-preprocess-gui` (clear & direct; current default suggestion)
- `nerf-dataset-prep`
- `nerf-cropper`
- `multiview-foreground-crop`
- `nerf-view-prep`
- `gaussian-prep-gui`
- `nerf-seg-crop`
- `nerf-capture-prep`

If emphasizing AI segmentation: `nerf-ai-cropper` or `nerf-rembg-prep`. If future scope broadens beyond cropping (calibration, exposure normalization), consider a more general umbrella like `nerf-dataset-toolkit`.

Recommended Choice: **`nerf-preprocess-gui`** (balanced clarity + brevity + GUI emphasis). Use that for initial release tag (e.g., `v0.1.0`).

## Roadmap Ideas

- Simple Mode (hide most rows until Advanced is expanded)
- Auto-switch preset label to "Custom" when manual changes occur
- Persist last settings in a JSON config
- Batch statistics (average crop area, timings)
- GPU acceleration path (optional)
- CLI exposure of recursive + pattern arguments
- Exclusion patterns (e.g. `--exclude processed_*;temp`) for discovery

## Troubleshooting

| Issue                    | Cause                                                       | Fix                                                      |
| ------------------------ | ----------------------------------------------------------- | -------------------------------------------------------- |
| Scenes show 0 images     | Root pointed at object folder? capture_* selected as scenes | Now supported; if still 0, verify pattern `cam_*.tiff` |
| Everything grayscale     | Source TIFF single channel                                  | Try Bayer pattern or confirm camera output               |
| Model download every run | Cache not persisted                                         | Ensure `%USERPROFILE%\\.u2net` writable                |
| Slow first frame         | AI model warm-up                                            | Subsequent frames faster                                 |

## License

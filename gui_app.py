import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

import cv2
from PyQt6 import QtWidgets, QtCore, QtGui

# Import preprocessing functions by module import
import preprocess as pp

@dataclass
class GUISettings:
    input_root: Path
    output_root: Path
    scenes: List[str]
    flat_mode: bool = False  # if True, process images directly from input_root
    pattern: str = 'cam_*.tiff'
    patterns_raw: str = ''  # semicolon separated patterns; empty => any supported image extensions
    recursive: bool = False
    channel: str = 'auto'
    method: str = 'otsu'
    percentile: float = 0.995
    invert: bool = False
    blur: int = 3
    morph_kernel: int = 5
    dilate: int = 2
    erode: int = 0
    no_keep_largest: bool = False
    pad: int = 8
    pad_rel: float = 0.0
    square: bool = False
    union_bbox: bool = False
    save_mask: bool = True
    save_rgba: bool = True
    save_cropped: bool = False
    bbox_only: bool = False
    rgba_opaque: bool = False
    enable_fallback: bool = False
    debug_visual: bool = True
    ai_seg: bool = True
    ai_seg_model: str = 'u2net'
    bayer_pattern: Optional[str] = 'rg'
    max_images: int = 0

class LogEmitter(QtCore.QObject):
    message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)  # current, total
    finished = QtCore.pyqtSignal(bool, str)

class Worker(QtCore.QObject):
    def __init__(self, settings: GUISettings, emitter: LogEmitter):
        super().__init__()
        self.settings = settings
        self.emitter = emitter
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            input_root = self.settings.input_root
            scenes = self.settings.scenes
            flat_mode = self.settings.flat_mode
            total = 0

            class Args:  # lightweight adapter passed into existing functions
                pass
            args = Args()
            for k,v in self.settings.__dict__.items():
                setattr(args, k, v)
            # Align attribute names expected by preprocess
            setattr(args, 'output_root', self.settings.output_root)
            setattr(args, 'input_root', self.settings.input_root)
            # Flags compatibility
            setattr(args, 'no_keep_largest', self.settings.no_keep_largest)

            if flat_mode:
                # Flat mode: process all images from input_root directly
                self.emitter.message.emit('Flat mode: processing images from root directory...')
                img_paths = pp.gather_images(
                    input_root,
                    pattern=self.settings.pattern,
                    patterns_raw=self.settings.patterns_raw,
                    recursive=self.settings.recursive,
                )
                total = len(img_paths)
                self.emitter.message.emit(f"Found {total} images in root directory")

                global_bbox = None
                if self.settings.union_bbox and img_paths:
                    self.emitter.message.emit(f"Computing global union bbox for ALL images ({len(img_paths)} total)...")
                    global_bbox = pp.compute_union_bbox(img_paths, args)
                    self.emitter.message.emit(f"Global bbox: {global_bbox}")

                metas = []
                out_dir = self.settings.output_root
                out_dir.mkdir(parents=True, exist_ok=True)
                processed = 0
                for p in img_paths:
                    if self._cancel:
                        self.emitter.message.emit('Cancellation requested... stopping.')
                        break
                    meta = pp.process_image(p, out_dir, args, global_bbox=global_bbox)
                    metas.append(meta)
                    processed += 1
                    self.emitter.progress.emit(processed, total)
                with open(out_dir / 'metadata.json', 'w') as f:
                    import json
                    json.dump({'images': metas, 'union_bbox': global_bbox}, f, indent=2)
                self.emitter.message.emit(f"Processed {len(metas)} images")
            else:
                # Scene mode: organize by subfolders
                # Gather all image paths from ALL scenes first
                all_img_paths = []
                scene_img_map = {}

                for s in scenes:
                    scene_dir = input_root / s
                    imgs = pp.gather_images(
                        scene_dir,
                        pattern=self.settings.pattern,
                        patterns_raw=self.settings.patterns_raw,
                        recursive=self.settings.recursive,
                    )
                    scene_img_map[s] = imgs
                    all_img_paths.extend(imgs)
                    total += len(imgs)

                # Compute GLOBAL bbox across ALL scenes if union_bbox is enabled
                global_bbox = None
                if self.settings.union_bbox and all_img_paths:
                    self.emitter.message.emit(f"Computing global union bbox across ALL scenes ({len(all_img_paths)} total images)...")
                    global_bbox = pp.compute_union_bbox(all_img_paths, args)
                    self.emitter.message.emit(f"Global bbox: {global_bbox}")

                processed = 0
                for scene_name in scenes:
                    if self._cancel:
                        self.emitter.message.emit('Cancelled before scene '+scene_name)
                        break
                    scene_dir = input_root / scene_name
                    if not scene_dir.is_dir():
                        self.emitter.message.emit(f"Skip missing scene {scene_name}")
                        continue

                    img_paths = scene_img_map.get(scene_name, [])
                    metas = []
                    out_dir = self.settings.output_root / scene_name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    for p in img_paths:
                        if self._cancel:
                            self.emitter.message.emit('Cancellation requested... stopping.')
                            break
                        # Use the global bbox for all scenes
                        meta = pp.process_image(p, out_dir, args, global_bbox=global_bbox)
                        metas.append(meta)
                        processed += 1
                        self.emitter.progress.emit(processed, total)

                    with open(out_dir / 'metadata.json', 'w') as f:
                        import json
                        json.dump({'scene': scene_name, 'images': metas, 'union_bbox': global_bbox}, f, indent=2)
                    self.emitter.message.emit(f"Scene {scene_name} done: {len(metas)} images")

            if self._cancel:
                self.emitter.finished.emit(False, 'Cancelled')
            else:
                self.emitter.finished.emit(True, 'Completed')
        except Exception as e:
            tb = traceback.format_exc()
            self.emitter.message.emit(tb)
            self.emitter.finished.emit(False, f'Error: {e}')

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Background Removal & Image Processor')
        self.resize(1100, 700)
        self.setMinimumSize(800, 600)
        self.emitter = LogEmitter()
        self.worker_thread = None
        self.worker = None
        self._use_custom_style = True  # start with modern custom style
        self._detected_mode = None  # 'flat', 'scene', or None
        self._build_ui()
        self._connect_signals()
        self._apply_style()  # applies modern style
        # Initial setup
        self._auto_output_dir()
        self._toggle_ai_deps()
        # Hide advanced options by default
        for label, widget in self._advanced_labels:
            label.setVisible(False)
            widget.setVisible(False)

    def _build_ui(self):
        # Create main scroll area for responsive layout
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_widget = QtWidgets.QWidget()
        scroll.setWidget(scroll_widget)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        layout = QtWidgets.QVBoxLayout(scroll_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Paths group - compact
        paths_group = QtWidgets.QGroupBox('📁 Input & Output')
        pg_layout = QtWidgets.QGridLayout(paths_group)
        pg_layout.setSpacing(6)
        pg_layout.setContentsMargins(8, 12, 8, 8)
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText('Select your input folder...')
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setReadOnly(True)
        in_btn = QtWidgets.QPushButton('Browse')
        in_btn.setMaximumWidth(80)
        out_btn = QtWidgets.QPushButton('↻')
        out_btn.setMaximumWidth(40)
        out_btn.setToolTip('Regenerate output folder name')
        in_btn.clicked.connect(lambda: self._pick_dir(self.input_edit))
        out_btn.clicked.connect(self._auto_output_dir)
        self.input_edit.textChanged.connect(self._on_input_changed)
        pg_layout.addWidget(QtWidgets.QLabel('Input:'), 0, 0)
        pg_layout.addWidget(self.input_edit, 0, 1)
        pg_layout.addWidget(in_btn, 0, 2)
        pg_layout.addWidget(QtWidgets.QLabel('Output:'), 1, 0)
        pg_layout.addWidget(self.output_edit, 1, 1)
        pg_layout.addWidget(out_btn, 1, 2)
        layout.addWidget(paths_group)

        # Image source info - compact
        self.source_info_group = QtWidgets.QGroupBox('🖼️ Detected Images')
        info_layout = QtWidgets.QVBoxLayout(self.source_info_group)
        info_layout.setSpacing(6)
        info_layout.setContentsMargins(8, 12, 8, 8)
        self.source_info_label = QtWidgets.QLabel('Select input folder to detect images...')
        self.source_info_label.setWordWrap(True)
        self.source_info_label.setStyleSheet('padding: 4px; color: #666; font-style: italic;')
        info_layout.addWidget(self.source_info_label)

        # Scene list (will be shown if scenes detected)
        self.scene_list = QtWidgets.QListWidget()
        self.scene_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.scene_list.setMaximumHeight(120)
        self.scene_list.setVisible(False)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.select_all_btn = QtWidgets.QPushButton('Select All')
        self.select_all_btn.setMaximumWidth(100)
        self.select_all_btn.setVisible(False)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addStretch()
        info_layout.addLayout(btn_row)
        info_layout.addWidget(self.scene_list)
        layout.addWidget(self.source_info_group)

        # Live Preview Section
        preview_group = QtWidgets.QGroupBox('👁️ Live Preview')
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setSpacing(6)
        preview_layout.setContentsMargins(8, 12, 8, 8)

        # Preview controls
        preview_controls = QtWidgets.QHBoxLayout()
        preview_controls.setSpacing(6)
        self.preview_btn = QtWidgets.QPushButton('🔍 Generate Preview')
        self.preview_btn.setToolTip('Preview detection on a random sample image')
        self.preview_btn.clicked.connect(self._generate_preview)
        preview_controls.addWidget(self.preview_btn)
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)

        # Preview image display
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setMaximumHeight(300)
        self.preview_label.setStyleSheet('QLabel { background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d; }')
        self.preview_label.setText('Click "Generate Preview" to see detection result')
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group)

        # Processing Options - compact
        opts_group = QtWidgets.QGroupBox('⚙️ Processing Options')
        og_layout = QtWidgets.QGridLayout(opts_group)
        og_layout.setSpacing(6)
        og_layout.setContentsMargins(8, 12, 8, 8)
        og_layout.setColumnStretch(1, 1)
        row = 0

        # Store label-widget pairs for advanced section
        self._advanced_labels = []

        def add_row(lbl, widget, is_advanced=False):
            nonlocal row
            label = QtWidgets.QLabel(lbl + ':')
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            label.setMinimumWidth(120)
            og_layout.addWidget(label, row, 0)
            og_layout.addWidget(widget, row, 1)
            if is_advanced:
                self._advanced_labels.append((label, widget))
            row += 1

        # Create all widgets first
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            'Custom',
            'AI Tight RGBA (transparent)',
            'AI Tight Crop (RGB only)',
            'AI Opaque RGBA',
            'Classical Threshold Crop',
            'BBox Only Fast',
        ])

        self.ai_seg_cb = QtWidgets.QCheckBox()
        self.ai_seg_cb.setChecked(True)
        self.ai_seg_cb.setToolTip('High quality background removal using rembg')

        # Padding mode - mutually exclusive
        self.pad_mode_absolute = QtWidgets.QRadioButton('Absolute (pixels)')
        self.pad_mode_relative = QtWidgets.QRadioButton('Relative (%)')
        self.pad_mode_relative.setChecked(True)

        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 400)
        self.pad_spin.setValue(8)
        self.pad_spin.setSuffix(' px')
        self.pad_spin.setToolTip('Extra pixels around detected object')
        self.pad_spin.setEnabled(False)

        self.pad_rel_spin = QtWidgets.QSpinBox()
        self.pad_rel_spin.setRange(0, 100)
        self.pad_rel_spin.setValue(20)
        self.pad_rel_spin.setSuffix(' %')
        self.pad_rel_spin.setToolTip('Relative padding as % of object size')

        # Connect radio buttons to enable/disable spinboxes and update preview
        self.pad_mode_absolute.toggled.connect(lambda checked: self.pad_spin.setEnabled(checked))
        self.pad_mode_relative.toggled.connect(lambda checked: self.pad_rel_spin.setEnabled(checked))
        self.pad_mode_absolute.toggled.connect(self._on_preview_settings_changed)
        self.pad_spin.valueChanged.connect(self._on_preview_settings_changed)
        self.pad_rel_spin.valueChanged.connect(self._on_preview_settings_changed)

        self.union_cb = QtWidgets.QCheckBox()
        self.union_cb.setToolTip('Use single bounding box computed from ALL images across all scenes (ensures globally consistent size)')

        self.square_cb = QtWidgets.QCheckBox()
        self.square_cb.setToolTip('Force square output (1:1 aspect ratio)')

        self.save_mask_cb = QtWidgets.QCheckBox()
        self.save_mask_cb.setChecked(True)
        self.save_mask_cb.setToolTip('Save binary mask (foreground/background)')

        self.save_rgba_cb = QtWidgets.QCheckBox()
        self.save_rgba_cb.setChecked(True)
        self.save_rgba_cb.setToolTip('Save image with transparency (PNG)')

        self.rgba_opaque_cb = QtWidgets.QCheckBox()
        self.rgba_opaque_cb.setToolTip('Save RGBA without transparency (for compatibility)')

        self.save_crop_cb = QtWidgets.QCheckBox()
        self.save_crop_cb.setToolTip('Save cropped RGB image')

        self.bbox_only_cb = QtWidgets.QCheckBox()
        self.bbox_only_cb.setToolTip('Fast mode: crop only, skip mask/RGBA')

        self.debug_cb = QtWidgets.QCheckBox()
        self.debug_cb.setChecked(True)
        self.debug_cb.setToolTip('Save visualization showing detected mask and crop')

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(['otsu', 'percentile'])
        self.method_combo.setToolTip('Classical segmentation method (when AI disabled)')

        self.percentile_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_spin.setRange(0.9, 0.9999)
        self.percentile_spin.setDecimals(4)
        self.percentile_spin.setValue(0.995)
        self.percentile_spin.setToolTip('Threshold value for percentile method')

        self.bayer_combo = QtWidgets.QComboBox()
        self.bayer_combo.addItems(['Auto/None', 'rg', 'gr', 'bg', 'gb'])
        self.bayer_combo.setCurrentText('rg')
        self.bayer_combo.setToolTip('Demosaic pattern for raw camera data')

        self.recursive_cb = QtWidgets.QCheckBox()
        self.recursive_cb.setToolTip('Search all subfolders recursively')

        self.patterns_edit = QtWidgets.QLineEdit()
        self.patterns_edit.setPlaceholderText('e.g. *.png;*.jpg (empty = all images)')
        self.patterns_edit.setToolTip('Filter files by pattern')


        # Add main options
        add_row('Preset', self.preset_combo)
        add_row('AI Segmentation', self.ai_seg_cb)

        # Padding mode selector
        pad_mode_widget = QtWidgets.QWidget()
        pad_mode_layout = QtWidgets.QHBoxLayout(pad_mode_widget)
        pad_mode_layout.setContentsMargins(0, 0, 0, 0)
        pad_mode_layout.addWidget(self.pad_mode_absolute)
        pad_mode_layout.addWidget(self.pad_mode_relative)
        pad_mode_layout.addStretch()
        add_row('Padding Mode', pad_mode_widget)

        add_row('Absolute Padding', self.pad_spin)
        add_row('Relative Padding', self.pad_rel_spin)
        add_row('Consistent Size (Global)', self.union_cb)
        add_row('Force Square', self.square_cb)
        add_row('Save Mask', self.save_mask_cb)
        add_row('Save RGBA', self.save_rgba_cb)
        add_row('Save Cropped RGB', self.save_crop_cb)
        add_row('Debug Visualization', self.debug_cb)

        # Advanced section toggle button
        self.adv_toggle = QtWidgets.QPushButton('▼ Show Advanced Options')
        self.adv_toggle.setCheckable(True)
        og_layout.addWidget(self.adv_toggle, row, 0, 1, 2)
        row += 1

        # Advanced options (initially hidden)
        add_row('RGBA Opaque Mode', self.rgba_opaque_cb, is_advanced=True)
        add_row('BBox Only (Fast)', self.bbox_only_cb, is_advanced=True)
        add_row('Mask Method', self.method_combo, is_advanced=True)
        add_row('Percentile Threshold', self.percentile_spin, is_advanced=True)
        add_row('Demosaic (Bayer)', self.bayer_combo, is_advanced=True)
        add_row('Recursive Search', self.recursive_cb, is_advanced=True)
        add_row('Filename Patterns', self.patterns_edit, is_advanced=True)

        # Connect signals
        self.adv_toggle.toggled.connect(self._toggle_advanced)
        self.ai_seg_cb.toggled.connect(self._toggle_ai_deps)
        self.ai_seg_cb.toggled.connect(self._on_preview_settings_changed)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        self.square_cb.toggled.connect(self._on_preview_settings_changed)
        self.union_cb.toggled.connect(self._on_preview_settings_changed)

        layout.addWidget(opts_group)

        # Run controls - modern compact
        run_group = QtWidgets.QGroupBox('▶️ Run')
        run_group_layout = QtWidgets.QVBoxLayout(run_group)
        run_group_layout.setSpacing(6)
        run_group_layout.setContentsMargins(8, 12, 8, 8)

        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(6)
        self.run_btn = QtWidgets.QPushButton('▶ Start Processing')
        self.run_btn.setMinimumHeight(40)
        self.cancel_btn = QtWidgets.QPushButton('⏹ Cancel')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMaximumWidth(100)
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.cancel_btn, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(24)

        self.status_label = QtWidgets.QLabel('Ready')
        self.status_label.setStyleSheet('padding: 4px; font-weight: bold;')

        run_group_layout.addLayout(run_row)
        run_group_layout.addWidget(self.progress)
        run_group_layout.addWidget(self.status_label)
        layout.addWidget(run_group)

        # Log area - collapsible
        log_group = QtWidgets.QGroupBox('📋 Processing Log')
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setSpacing(6)
        log_layout.setContentsMargins(8, 12, 8, 8)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group)

        # Button connections
        self.select_all_btn.clicked.connect(self._select_all_scenes)
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)

    def _pick_dir(self, line_edit: QtWidgets.QLineEdit):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if d:
            line_edit.setText(d)

    def _on_input_changed(self):
        """Called when input folder text changes."""
        self._auto_output_dir()
        # Debounce: only detect if path exists
        root = self.input_edit.text().strip()
        if root and Path(root).is_dir():
            self._detect_images()
            # Clear preview when input changes
            self._preview_image_path = None
            self.preview_label.clear()
            self.preview_label.setText('Click "Generate Preview" to see detection result')

    def _on_preview_settings_changed(self):
        """Called when any preview-affecting setting changes."""
        # Auto-regenerate preview if we have a cached image
        if hasattr(self, '_preview_image_path') and self._preview_image_path:
            self._generate_preview()

    def _detect_images(self):
        """Automatically detect images and scenes intelligently."""
        root = self.input_edit.text().strip()
        if not root:
            return
        p = Path(root)
        if not p.is_dir():
            return

        # Try to detect images at root level
        root_images = pp.gather_images(p, patterns_raw='', recursive=False)

        # Try to detect subdirectories (potential scenes)
        subdirs = [sub for sub in sorted(p.iterdir()) if sub.is_dir()]

        # Intelligent detection
        if len(root_images) > 0 and len(subdirs) == 0:
            # Case 1: Images at root, no subdirectories -> Flat mode
            self._detected_mode = 'flat'
            self.source_info_label.setText(f'✓ Found {len(root_images)} images in root directory (flat mode)')
            self.source_info_label.setStyleSheet('color: #2e7d32; font-weight: bold;')
            self.scene_list.setVisible(False)
            self.select_all_btn.setVisible(False)

        elif len(subdirs) > 0:
            # Case 2: Has subdirectories -> Check if they contain images (scene mode)
            scene_count = 0
            self.scene_list.clear()
            for sub in subdirs:
                sub_images = pp.gather_images(sub, patterns_raw='', recursive=False)
                if len(sub_images) > 0:
                    self.scene_list.addItem(sub.name)
                    scene_count += 1

            if scene_count > 0:
                # Scene mode: subdirectories with images
                self._detected_mode = 'scene'
                self.source_info_label.setText(f'✓ Found {scene_count} scene(s) with images (select scenes to process)')
                self.source_info_label.setStyleSheet('color: #1976d2; font-weight: bold;')
                self.scene_list.setVisible(True)
                self.select_all_btn.setVisible(True)
                # Auto-select all scenes
                for i in range(self.scene_list.count()):
                    self.scene_list.item(i).setSelected(True)
            elif len(root_images) > 0:
                # Has subdirs but they're empty, but root has images -> Flat mode
                self._detected_mode = 'flat'
                self.source_info_label.setText(f'✓ Found {len(root_images)} images in root directory (flat mode)')
                self.source_info_label.setStyleSheet('color: #2e7d32; font-weight: bold;')
                self.scene_list.setVisible(False)
                self.select_all_btn.setVisible(False)
            else:
                # No images found anywhere
                self._detected_mode = None
                self.source_info_label.setText('⚠ No images found in directory or subdirectories')
                self.source_info_label.setStyleSheet('color: #d32f2f; font-weight: bold;')
                self.scene_list.setVisible(False)
                self.select_all_btn.setVisible(False)
        else:
            # No images, no subdirs
            self._detected_mode = None
            self.source_info_label.setText('⚠ No images or subdirectories found')
            self.source_info_label.setStyleSheet('color: #d32f2f; font-weight: bold;')
            self.scene_list.setVisible(False)
            self.select_all_btn.setVisible(False)

    def _select_all_scenes(self):
        for i in range(self.scene_list.count()):
            item = self.scene_list.item(i)
            item.setSelected(True)

    def _generate_preview(self):
        """Generate a preview of the detection on a sample image."""
        import random

        root = self.input_edit.text().strip()
        if not root or not Path(root).is_dir():
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please select a valid input folder first')
            return

        try:
            # Get sample image
            if not hasattr(self, '_preview_image_path') or not self._preview_image_path:
                # Pick a random image from detected images
                if self._detected_mode == 'flat':
                    img_paths = pp.gather_images(Path(root), patterns_raw='', recursive=False)
                elif self._detected_mode == 'scene':
                    # Pick from first selected scene or first scene
                    selected = [i.text() for i in self.scene_list.selectedItems()]
                    if selected:
                        scene_dir = Path(root) / selected[0]
                    else:
                        # Pick first scene
                        scene_dir = Path(root) / self.scene_list.item(0).text()
                    img_paths = pp.gather_images(scene_dir, patterns_raw='', recursive=False)
                else:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'No images detected')
                    return

                if not img_paths:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'No images found')
                    return

                self._preview_image_path = random.choice(img_paths)

            # Load image
            img = pp.load_image(self._preview_image_path, None)

            # Get current settings
            use_ai = self.ai_seg_cb.isChecked()

            # Compute mask
            if use_ai:
                if not pp.REMBG_AVAILABLE:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'rembg not installed. Install with: pip install rembg')
                    return
                mask = pp.ai_segment(img, 'u2net')
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                mask = pp.compute_foreground_mask(img, channel='auto', method='otsu', invert=True, blur=3)

            mask = pp.refine_mask(mask, morph_kernel=5, dilate=2, erode=0, keep_largest=True)

            # Calculate padding
            if self.pad_mode_absolute.isChecked():
                pad_px = self.pad_spin.value()
                pad_rel = 0.0
            else:
                pad_px = 0
                pad_rel = self.pad_rel_spin.value() / 100.0

            # Get bbox
            bbox = pp.mask_to_bbox(mask, pad=pad_px, square=self.square_cb.isChecked(), pad_rel=pad_rel)

            # Draw preview
            if img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw bbox
            x0, y0, x1, y1 = bbox
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 0), 3)

            # Add text
            text = f"BBox: {x1-x0}x{y1-y0} px"
            cv2.putText(rgb, text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Scale to fit preview
            h, w = rgb.shape[:2]
            max_h = 280
            max_w = self.preview_label.width() - 20
            scale = min(max_h / h, max_w / w, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to QPixmap
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)

            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet('QLabel { background: white; border: 2px solid #0d6efd; border-radius: 8px; }')

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            QtWidgets.QMessageBox.critical(self, 'Preview Error', f'Failed to generate preview:\n{e}\n\n{tb}')
            self.preview_label.setText(f'Preview failed: {e}')

    def _collect_settings(self) -> Optional[GUISettings]:
        try:
            input_root = Path(self.input_edit.text().strip())
            output_root = Path(self.output_edit.text().strip())
            if not input_root.is_dir():
                QtWidgets.QMessageBox.warning(self, 'Error', 'Input root invalid')
                return None
            if not output_root.exists():
                output_root.mkdir(parents=True, exist_ok=True)

            # Use detected mode
            if not hasattr(self, '_detected_mode') or self._detected_mode is None:
                QtWidgets.QMessageBox.warning(self, 'Error', 'No images detected. Please select a valid input folder.')
                return None

            flat_mode = (self._detected_mode == 'flat')
            scenes = []
            if not flat_mode:
                scenes = [i.text() for i in self.scene_list.selectedItems()]
                if not scenes:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'Select at least one scene')
                    return None

            # Use only active padding mode
            if self.pad_mode_absolute.isChecked():
                pad_px = self.pad_spin.value()
                pad_rel = 0.0
            else:
                pad_px = 0
                pad_rel = self.pad_rel_spin.value() / 100.0

            bayer_val = self.bayer_combo.currentText()
            if bayer_val == 'Auto/None':
                bayer_val = None
            return GUISettings(
                input_root=input_root,
                output_root=output_root,
                scenes=scenes,
                flat_mode=flat_mode,
                method=self.method_combo.currentText(),
                percentile=self.percentile_spin.value(),
                pad=pad_px,
                pad_rel=pad_rel,
                union_bbox=self.union_cb.isChecked(),
                square=self.square_cb.isChecked(),
                save_mask=self.save_mask_cb.isChecked(),
                save_rgba=self.save_rgba_cb.isChecked(),
                save_cropped=self.save_crop_cb.isChecked(),
                bbox_only=self.bbox_only_cb.isChecked(),
                rgba_opaque=self.rgba_opaque_cb.isChecked(),
                debug_visual=self.debug_cb.isChecked(),
                ai_seg=self.ai_seg_cb.isChecked(),
                patterns_raw=self.patterns_edit.text().strip(),
                recursive=self.recursive_cb.isChecked(),
                bayer_pattern=bayer_val,
                max_images=0,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'Failed collect settings: {e}')
            return None

    def _start(self):
        settings = self._collect_settings()
        if not settings:
            return
        self.log_edit.clear()
        self.progress.setValue(0)
        self.status_label.setText('Running...')
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Disable all settings during processing
        self._set_settings_enabled(False)

        self.worker = Worker(settings, self.emitter)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText('Cancelling...')

    def _connect_signals(self):
        self.emitter.message.connect(self._on_message)
        self.emitter.progress.connect(self._on_progress)
        self.emitter.finished.connect(self._on_finished)

    def _on_message(self, msg: str):
        self.log_edit.appendPlainText(msg)

    def _on_progress(self, cur: int, total: int):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(cur)
            self.status_label.setText(f"{cur}/{total}")

    def _on_finished(self, ok: bool, msg: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(msg)
        self.log_edit.appendPlainText(f"Finished: {msg}")

        # Re-enable all settings after processing
        self._set_settings_enabled(True)

    # --- Helpers --- #
    def _set_settings_enabled(self, enabled: bool):
        """Enable or disable all settings widgets during processing."""
        # Input controls
        self.input_edit.setEnabled(enabled)
        self.preview_btn.setEnabled(enabled)
        self.select_all_btn.setEnabled(enabled)
        self.scene_list.setEnabled(enabled)

        # Processing options
        self.preset_combo.setEnabled(enabled)
        self.ai_seg_cb.setEnabled(enabled)
        self.pad_mode_absolute.setEnabled(enabled)
        self.pad_mode_relative.setEnabled(enabled)
        self.pad_spin.setEnabled(enabled and self.pad_mode_absolute.isChecked())
        self.pad_rel_spin.setEnabled(enabled and self.pad_mode_relative.isChecked())
        self.union_cb.setEnabled(enabled)
        self.square_cb.setEnabled(enabled)
        self.save_mask_cb.setEnabled(enabled)
        self.save_rgba_cb.setEnabled(enabled)
        self.save_crop_cb.setEnabled(enabled)
        self.debug_cb.setEnabled(enabled)

        # Advanced options
        self.rgba_opaque_cb.setEnabled(enabled)
        self.bbox_only_cb.setEnabled(enabled)
        self.method_combo.setEnabled(enabled)
        self.percentile_spin.setEnabled(enabled)
        self.bayer_combo.setEnabled(enabled)
        self.recursive_cb.setEnabled(enabled)
        self.patterns_edit.setEnabled(enabled)

    def _auto_output_dir(self):
        root = self.input_edit.text().strip()
        if not root:
            return
        base = Path(root)
        if not base.is_dir():
            return
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f'processed_{ts}'
        self.output_edit.setText(str(base / out_name))

    def _toggle_advanced(self):
        show = self.adv_toggle.isChecked()
        self.adv_toggle.setText('▲ Hide Advanced Options' if show else '▼ Show Advanced Options')
        for label, widget in self._advanced_labels:
            label.setVisible(show)
            widget.setVisible(show)

    def _toggle_ai_deps(self):
        ai_on = self.ai_seg_cb.isChecked()
        self.method_combo.setEnabled(not ai_on)
        self.percentile_spin.setEnabled(not ai_on and self.method_combo.currentText() == 'percentile')
        self.method_combo.setToolTip('Disabled when AI segmentation is active.')

    def _apply_preset(self, name: str):
        if name == 'Custom':
            return
        # Snapshot: prevent recursive signals
        block = [self.ai_seg_cb, self.save_mask_cb, self.save_rgba_cb, self.rgba_opaque_cb,
                 self.save_crop_cb, self.bbox_only_cb, self.square_cb, self.union_cb]
        for w in block:
            w.blockSignals(True)
        try:
            if name.startswith('AI Tight RGBA'):
                self.ai_seg_cb.setChecked(True)
                self.save_mask_cb.setChecked(True)
                self.save_rgba_cb.setChecked(True)
                self.rgba_opaque_cb.setChecked(False)
                self.save_crop_cb.setChecked(False)
                self.bbox_only_cb.setChecked(False)
                self.pad_spin.setValue(4)
                self.pad_rel_spin.setValue(10)
                self.union_cb.setChecked(False)
                self.square_cb.setChecked(False)
            elif name.startswith('AI Tight Crop'):
                self.ai_seg_cb.setChecked(True)
                self.save_mask_cb.setChecked(False)
                self.save_rgba_cb.setChecked(False)
                self.rgba_opaque_cb.setChecked(False)
                self.save_crop_cb.setChecked(True)
                self.bbox_only_cb.setChecked(True)
                self.pad_spin.setValue(4)
                self.pad_rel_spin.setValue(8)
                self.union_cb.setChecked(False)
                self.square_cb.setChecked(False)
            elif name == 'AI Opaque RGBA':
                self.ai_seg_cb.setChecked(True)
                self.save_mask_cb.setChecked(False)
                self.save_rgba_cb.setChecked(True)
                self.rgba_opaque_cb.setChecked(True)
                self.save_crop_cb.setChecked(False)
                self.bbox_only_cb.setChecked(False)
                self.pad_spin.setValue(8)
                self.pad_rel_spin.setValue(15)
                self.union_cb.setChecked(False)
                self.square_cb.setChecked(False)
            elif name == 'Classical Threshold Crop':
                self.ai_seg_cb.setChecked(False)
                self.method_combo.setCurrentText('otsu')
                self.save_mask_cb.setChecked(True)
                self.save_rgba_cb.setChecked(True)
                self.rgba_opaque_cb.setChecked(False)
                self.save_crop_cb.setChecked(False)
                self.bbox_only_cb.setChecked(False)
                self.pad_spin.setValue(12)
                self.pad_rel_spin.setValue(20)
                self.union_cb.setChecked(False)
                self.square_cb.setChecked(False)
            elif name == 'BBox Only Fast':
                self.ai_seg_cb.setChecked(True)
                self.save_mask_cb.setChecked(False)
                self.save_rgba_cb.setChecked(False)
                self.rgba_opaque_cb.setChecked(False)
                self.save_crop_cb.setChecked(True)
                self.bbox_only_cb.setChecked(True)
                self.pad_spin.setValue(6)
                self.pad_rel_spin.setValue(12)
                self.union_cb.setChecked(False)
                self.square_cb.setChecked(False)
        finally:
            for w in block:
                w.blockSignals(False)
        # Reflect that we applied something customizable; user can tweak further.
        self._toggle_ai_deps()

    def _apply_style(self):
        if not self._use_custom_style:
            self.setStyleSheet("")
            return

        # Modern gradient style - always light theme
        self.setStyleSheet('''
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 13px;
                color: #212529;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 13px;
                border: 1px solid #dee2e6;
                border-radius: 12px;
                margin-top: 12px;
                padding: 12px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background: white;
            }
            QPushButton {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 8px 16px;
                color: #212529;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f8f9fa;
                border-color: #0d6efd;
            }
            QPushButton:pressed {
                background: #e9ecef;
            }
            QPushButton:disabled {
                color: #adb5bd;
                background: #f8f9fa;
                border-color: #dee2e6;
            }
            QPushButton#run_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d6efd, stop:1 #0a58ca);
                color: white;
                border: none;
            }
            QPushButton#run_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a58ca, stop:1 #084298);
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: white;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 6px 10px;
                color: #212529;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #0d6efd;
                padding: 5px 9px;
            }
            QLineEdit:read-only {
                background: #f8f9fa;
                color: #6c757d;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                text-align: center;
                background: white;
                color: #212529;
                font-weight: 500;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d6efd, stop:1 #0dcaf0);
                border-radius: 7px;
            }
            QPlainTextEdit {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                color: #212529;
                padding: 8px;
                font-family: "Consolas", "Monaco", monospace;
                font-size: 12px;
            }
            QListWidget {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 4px;
            }
            QListWidget::item {
                border-radius: 4px;
                padding: 6px 8px;
            }
            QListWidget::item:selected {
                background: #e7f1ff;
                color: #0d6efd;
            }
            QListWidget::item:hover {
                background: #f8f9fa;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #f8f9fa;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #adb5bd;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #ced4da;
                border-radius: 4px;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #0d6efd;
                border-color: #0d6efd;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgNEw0LjUgNy41TDExIDEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+);
            }
            QCheckBox::indicator:hover {
                border-color: #0d6efd;
            }
        ''')
        # Set object name for the run button to apply special styling
        self.run_btn.setObjectName('run_btn')

    def _toggle_theme(self):
        self._use_custom_style = not self._use_custom_style
        self.theme_btn.setText('System Theme' if self._use_custom_style else 'Custom Theme')
        self._apply_style()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

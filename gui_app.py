import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

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
                if self.settings.max_images > 0:
                    img_paths = img_paths[:self.settings.max_images]
                total = len(img_paths)
                self.emitter.message.emit(f"Found {total} images in root directory")

                global_bbox = None
                if self.settings.union_bbox and img_paths:
                    self.emitter.message.emit(f"Computing union bbox ({len(img_paths)} images)...")
                    global_bbox = pp.compute_union_bbox(img_paths, args)

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
                # gather images count for progress baseline
                for s in scenes:
                    scene_dir = input_root / s
                    imgs = pp.gather_images(
                        scene_dir,
                        pattern=self.settings.pattern,
                        patterns_raw=self.settings.patterns_raw,
                        recursive=self.settings.recursive,
                    )
                    if self.settings.max_images > 0:
                        imgs = imgs[:self.settings.max_images]
                    total += len(imgs)
                processed = 0

                for scene_name in scenes:
                    if self._cancel:
                        self.emitter.message.emit('Cancelled before scene '+scene_name)
                        break
                    scene_dir = input_root / scene_name
                    if not scene_dir.is_dir():
                        self.emitter.message.emit(f"Skip missing scene {scene_name}")
                        continue
                    img_paths = pp.gather_images(
                        scene_dir,
                        pattern=self.settings.pattern,
                        patterns_raw=self.settings.patterns_raw,
                        recursive=self.settings.recursive,
                    )
                    if self.settings.max_images > 0:
                        img_paths = img_paths[:self.settings.max_images]
                    global_bbox = None
                    if self.settings.union_bbox and img_paths:
                        self.emitter.message.emit(f"Computing union bbox for {scene_name} ({len(img_paths)} images)...")
                        global_bbox = pp.compute_union_bbox(img_paths, args)
                    metas = []
                    out_dir = self.settings.output_root / scene_name
                    out_dir.mkdir(parents=True, exist_ok=True)
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
        self.setWindowTitle('NeRF Preprocess')
        self.resize(980, 760)
        self.emitter = LogEmitter()
        self.worker_thread = None
        self.worker = None
        self._use_custom_style = False  # start with system style
        self._detected_mode = None  # 'flat', 'scene', or None
        self._build_ui()
        self._connect_signals()
        self._apply_style()  # applies (currently system/default)
        # Initial setup
        self._auto_output_dir()
        self._toggle_ai_deps()
        # Hide advanced options by default
        for label, widget in self._advanced_labels:
            label.setVisible(False)
            widget.setVisible(False)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Paths group
        paths_group = QtWidgets.QGroupBox('Input & Output')
        pg_layout = QtWidgets.QGridLayout(paths_group)
        pg_layout.setSpacing(8)
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText('Select your input folder...')
        self.output_edit = QtWidgets.QLineEdit(); self.output_edit.setReadOnly(True)
        in_btn = QtWidgets.QPushButton('Browse...')
        out_btn = QtWidgets.QPushButton('Regenerate')
        in_btn.clicked.connect(lambda: self._pick_dir(self.input_edit))
        out_btn.clicked.connect(self._auto_output_dir)
        self.input_edit.textChanged.connect(self._on_input_changed)
        pg_layout.addWidget(QtWidgets.QLabel('Input Folder:'), 0,0)
        pg_layout.addWidget(self.input_edit, 0,1)
        pg_layout.addWidget(in_btn, 0,2)
        pg_layout.addWidget(QtWidgets.QLabel('Output Folder:'), 1,0)
        pg_layout.addWidget(self.output_edit, 1,1)
        pg_layout.addWidget(out_btn, 1,2)
        layout.addWidget(paths_group)

        # Image source info
        self.source_info_group = QtWidgets.QGroupBox('Detected Images')
        info_layout = QtWidgets.QVBoxLayout(self.source_info_group)
        info_layout.setSpacing(8)
        self.source_info_label = QtWidgets.QLabel('Select input folder to detect images...')
        self.source_info_label.setWordWrap(True)
        self.source_info_label.setStyleSheet('color: #666; font-style: italic;')
        info_layout.addWidget(self.source_info_label)

        # Scene list (will be shown if scenes detected)
        self.scene_list = QtWidgets.QListWidget()
        self.scene_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.scene_list.setVisible(False)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        self.select_all_btn = QtWidgets.QPushButton('Select All')
        self.select_all_btn.setVisible(False)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addStretch()
        info_layout.addLayout(btn_row)
        info_layout.addWidget(self.scene_list)
        layout.addWidget(self.source_info_group)

        # Processing Options
        opts_group = QtWidgets.QGroupBox('Processing Options')
        og_layout = QtWidgets.QGridLayout(opts_group)
        og_layout.setSpacing(8)
        og_layout.setColumnStretch(1, 1)
        row = 0

        # Store label-widget pairs for advanced section
        self._advanced_labels = []

        def add_row(lbl, widget, is_advanced=False):
            nonlocal row
            label = QtWidgets.QLabel(lbl + ':')
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
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

        self.pad_spin = QtWidgets.QSpinBox()
        self.pad_spin.setRange(0, 400)
        self.pad_spin.setValue(8)
        self.pad_spin.setSuffix(' px')
        self.pad_spin.setToolTip('Extra pixels around detected object')

        self.pad_rel_spin = QtWidgets.QSpinBox()
        self.pad_rel_spin.setRange(0, 100)
        self.pad_rel_spin.setValue(20)
        self.pad_rel_spin.setSuffix(' %')
        self.pad_rel_spin.setToolTip('Relative padding as % of object size')

        self.union_cb = QtWidgets.QCheckBox()
        self.union_cb.setToolTip('Use same crop box for all images (consistent framing)')

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

        self.max_images_spin = QtWidgets.QSpinBox()
        self.max_images_spin.setRange(0, 100000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setToolTip('Limit processing (0 = all images)')

        # Add main options
        add_row('Preset', self.preset_combo)
        add_row('AI Segmentation', self.ai_seg_cb)
        add_row('Padding (absolute)', self.pad_spin)
        add_row('Padding (relative)', self.pad_rel_spin)
        add_row('Consistent Crop', self.union_cb)
        add_row('Force Square', self.square_cb)
        add_row('Save Mask', self.save_mask_cb)
        add_row('Save RGBA', self.save_rgba_cb)
        add_row('Save Cropped RGB', self.save_crop_cb)
        add_row('Debug Visualization', self.debug_cb)
        add_row('Limit Images', self.max_images_spin)

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
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        layout.addWidget(opts_group)

        # Run controls
        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(8)
        self.run_btn = QtWidgets.QPushButton('▶ Run Processing')
        self.run_btn.setMinimumHeight(36)
        font = self.run_btn.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        self.run_btn.setFont(font)
        self.cancel_btn = QtWidgets.QPushButton('Cancel'); self.cancel_btn.setEnabled(False)
        self.theme_btn = QtWidgets.QPushButton('Toggle Theme')
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.cancel_btn, 1)
        run_row.addWidget(self.theme_btn, 1)
        self.progress = QtWidgets.QProgressBar(); self.progress.setValue(0); self.progress.setTextVisible(True)
        layout.addLayout(run_row)
        layout.addWidget(self.progress)

        # Log area
        log_label = QtWidgets.QLabel('Processing Log:')
        layout.addWidget(log_label)
        self.log_edit = QtWidgets.QPlainTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        layout.addWidget(self.log_edit, stretch=1)

        # Footer / status
        self.status_label = QtWidgets.QLabel('Ready')
        self.status_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self.status_label)

        # Button connections
        self.select_all_btn.clicked.connect(self._select_all_scenes)
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        self.theme_btn.clicked.connect(self._toggle_theme)

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

            pad_rel_fraction = self.pad_rel_spin.value() / 100.0
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
                pad=self.pad_spin.value(),
                pad_rel=pad_rel_fraction,
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
                max_images=self.max_images_spin.value(),
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

    # --- Helpers --- #
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
            # Restore system style: clear stylesheet
            self.setStyleSheet("")
            pal = self.palette()
            # Ensure base contrast (if system is light, keep; if dark, trust system)
            self.setPalette(pal)
            return

        # Adaptive custom theme (light/dark based on system base color luminance)
        pal = self.palette()
        base_color = pal.color(pal.ColorRole.Base)
        # Simple luminance check
        luminance = (0.2126*base_color.redF() + 0.7152*base_color.greenF() + 0.0722*base_color.blueF())
        dark = luminance < 0.5
        if dark:
            bg0 = '#1F1F22'
            bg1 = '#2A2A2E'
            panel = '#242428'
            border = '#3A3A40'
            text = '#EDEDED'
            accent = '#4A90E2'
            accent_dim = '#2F6CB2'
            progress_chunk = accent
        else:
            bg0 = '#F4F4F6'
            bg1 = '#FFFFFF'
            panel = '#FFFFFF'
            border = '#D7D7DA'
            text = '#202124'
            accent = '#3578E5'
            accent_dim = '#2B63BD'
            progress_chunk = accent

        self.setStyleSheet(f'''
            QWidget {{ background: {bg0}; font-family: "Segoe UI", "Helvetica Neue", Arial; color:{text}; }}
            QGroupBox {{ font-weight:600; border:1px solid {border}; border-radius:10px; margin-top:8px; padding:8px 10px 10px 10px; background:{panel}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding:2px 4px; background: transparent; }}
            QPushButton {{ background:{bg1}; border:1px solid {border}; border-radius:10px; padding:6px 14px; color:{text}; }}
            QPushButton:hover {{ border-color:{accent}; }}
            QPushButton:pressed {{ background:{accent}22; }}
            QPushButton:disabled {{ color: #888; background:{bg0}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:{bg1}; border:1px solid {border}; border-radius:8px; padding:4px 6px; color:{text}; }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border:1px solid {accent}; }}
            QProgressBar {{ border:1px solid {border}; border-radius:10px; text-align:center; height:20px; background:{panel}; color:{text}; }}
            QProgressBar::chunk {{ background-color:{progress_chunk}; border-radius:10px; }}
            QPlainTextEdit {{ background:{bg1}; border:1px solid {border}; border-radius:10px; color:{text}; }}
            QListWidget {{ background:{bg1}; border:1px solid {border}; border-radius:8px; }}
            QLabel {{ color:{text}; }}
            QScrollBar:vertical {{ background: {panel}; width:12px; margin:2px; border-radius:6px; }}
            QScrollBar::handle:vertical {{ background:{border}; border-radius:6px; min-height:24px; }}
            QScrollBar::handle:vertical:hover {{ background:{accent}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
        ''')

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

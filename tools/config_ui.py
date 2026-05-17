"""
PySide6 config tuning UI for the leaflet angle pipeline.

Usage:
    python tools/config_ui.py [--data data]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QPushButton, QComboBox, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from pipeline.config_loader import load_config, save_config
from pipeline.leaflet_detection import preprocess_image, count_white_pixels, fit_circle


# ── rect metadata ────────────────────────────────────────────────────────────
RECTS = [
    ("laser_prim", "img_laser_prim_tl", "img_laser_prim_br", QColor(255, 80,  80),  "Laser primary"),
    ("laser",      "img_laser_tl",      "img_laser_br",      QColor(80,  200, 80),  "Laser"),
    ("leaflet",    "img_leafl_tl",      "img_leafl_br",      QColor(80,  160, 255), "Leaflet"),
]

CIRCLE_COLOR = QColor(255, 165, 0)   # orange — fitted circle
CIRCLE_PT_COLOR = QColor(255, 220, 0)  # yellow — selected points


def _cv2_to_pixmap(img):
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888))


class ImageCanvas(QLabel):
    """Displays a frame; click sets crop corners (rect mode) or circle points (circle mode)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self._cfg = {}
        self._mode = "rect"           # "rect" or "circle"
        self._active_rect = "laser_prim"
        self._click_count = 0
        self._base_pixmap = None
        self._mask_overlay = None
        # circle fit state (full-image coords)
        self._circle_pts = [None, None, None]
        self._active_circle_slot = 0
        self._circle_center = None   # (cx, cy) full-image
        self._circle_radius = None

    def set_cfg(self, cfg):
        self._cfg = cfg

    def set_mode(self, mode):
        self._mode = mode

    def set_active_rect(self, name):
        self._active_rect = name
        self._click_count = 0

    def set_active_circle_slot(self, idx):
        self._active_circle_slot = idx

    def set_circle_result(self, cx, cy, r):
        self._circle_center = (cx, cy)
        self._circle_radius = r
        self.update()

    def clear_circle_pts(self):
        self._circle_pts = [None, None, None]
        self._circle_center = None
        self._circle_radius = None
        self.update()

    def set_base_image(self, img_bgr):
        self._base_pixmap = _cv2_to_pixmap(img_bgr)
        self._mask_overlay = None
        self.update()

    def set_mask_overlay(self, mask_bgr):
        self._mask_overlay = mask_bgr
        self.update()

    def clear_mask(self):
        self._mask_overlay = None
        self.update()

    def mousePressEvent(self, event):
        if self._base_pixmap is None:
            return
        scale = self._display_scale()
        x = int(event.position().x() / scale)
        y = int(event.position().y() / scale)

        if self._mode == "rect":
            rect_info = next(r for r in RECTS if r[0] == self._active_rect)
            if self._click_count % 2 == 0:
                self._cfg[rect_info[1]] = [x, y]
            else:
                self._cfg[rect_info[2]] = [x, y]
            self._click_count += 1
            self.update()
            if hasattr(self.parent(), "_on_rect_changed"):
                self.parent()._on_rect_changed()
        else:  # circle mode
            self._circle_pts[self._active_circle_slot] = (x, y)
            self.update()
            if hasattr(self.parent(), "_on_circle_clicked"):
                self.parent()._on_circle_clicked(x, y, self._active_circle_slot)

    def _display_scale(self):
        if self._base_pixmap is None:
            return 1.0
        pw = self._base_pixmap.width()
        lw = self.width()
        return lw / pw if pw > 0 else 1.0

    def paintEvent(self, event):
        if self._base_pixmap is None:
            super().paintEvent(event)
            return
        scale = self._display_scale()
        w = int(self._base_pixmap.width()  * scale)
        h = int(self._base_pixmap.height() * scale)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, w, h, self._base_pixmap)

        # mask overlay (semi-transparent)
        if self._mask_overlay is not None:
            mask_px = _cv2_to_pixmap(self._mask_overlay)
            painter.setOpacity(0.4)
            painter.drawPixmap(0, 0, w, h, mask_px)
            painter.setOpacity(1.0)

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # draw crop rects
        for _, tl_key, br_key, color, label in RECTS:
            tl = self._cfg.get(tl_key)
            br = self._cfg.get(br_key)
            if tl and br:
                painter.setPen(QPen(color, 2))
                x1, y1 = int(tl[0] * scale), int(tl[1] * scale)
                x2, y2 = int(br[0] * scale), int(br[1] * scale)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1 + 3, y1 + 14, label)

        # draw circle points
        for i, pt in enumerate(self._circle_pts):
            if pt is not None:
                px, py = int(pt[0] * scale), int(pt[1] * scale)
                painter.setPen(QPen(CIRCLE_PT_COLOR, 2))
                arm = 10
                painter.drawLine(px - arm, py, px + arm, py)
                painter.drawLine(px, py - arm, px, py + arm)
                painter.drawText(px + 4, py - 4, str(i + 1))

        # draw fitted circle
        if self._circle_center is not None and self._circle_radius is not None:
            cx, cy = self._circle_center
            r = self._circle_radius
            painter.setPen(QPen(CIRCLE_COLOR, 2))
            painter.drawEllipse(QRectF(
                (cx - r) * scale, (cy - r) * scale,
                2 * r * scale, 2 * r * scale,
            ))
            arm = 8
            scx, scy = cx * scale, cy * scale
            painter.drawLine(int(scx - arm), int(scy), int(scx + arm), int(scy))
            painter.drawLine(int(scx), int(scy - arm), int(scx), int(scy + arm))
            painter.drawText(int(scx + 6), int(scy - 4), f"({cx:.0f}, {cy:.0f})")

        painter.end()


class ConfigWindow(QMainWindow):
    def __init__(self, data_dir):
        super().__init__()
        self.setWindowTitle("Pipeline Config Tuning")
        self.data_dir = data_dir
        self.cfg = {}
        self._image = None
        self._circle_frame_nums = [None, None, None]

        self._build_ui()
        self._populate_datasets()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        ds_row.addWidget(self.dataset_combo, 1)
        root.addLayout(ds_row)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)
        tabs.addTab(self._build_tab1(), "Crops & Thresholds")
        tabs.addTab(self._build_tab2(), "Ranges & Other")

        bar = QHBoxLayout()
        self.restore_btn = QPushButton("Restore default config")
        self.restore_btn.clicked.connect(self._restore_default)
        self.save_btn = QPushButton("Save custom config")
        self.save_btn.clicked.connect(self._save)
        bar.addStretch()
        bar.addWidget(self.restore_btn)
        bar.addWidget(self.save_btn)
        root.addLayout(bar)

    def _build_tab1(self):
        w = QWidget()
        layout = QHBoxLayout(w)

        self.canvas = ImageCanvas(w)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 3)

        ctrl = QVBoxLayout()
        layout.addLayout(ctrl, 1)

        # frame selector
        fr_grp = QGroupBox("Frame")
        fr_lay = QHBoxLayout(fr_grp)
        fr_lay.addWidget(QLabel("Frame #"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 99999)
        self.frame_spin.valueChanged.connect(self._load_frame)
        fr_lay.addWidget(self.frame_spin)
        ctrl.addWidget(fr_grp)

        # mode toggle
        mode_grp = QGroupBox("Click mode")
        mode_lay = QHBoxLayout(mode_grp)
        self._mode_btn_group = QButtonGroup(self)
        rb_rect   = QRadioButton("Crop rect")
        rb_circle = QRadioButton("Circle point")
        rb_rect.setChecked(True)
        rb_rect.toggled.connect(lambda checked: checked and self.canvas.set_mode("rect"))
        rb_circle.toggled.connect(lambda checked: checked and self.canvas.set_mode("circle"))
        self._mode_btn_group.addButton(rb_rect, 0)
        self._mode_btn_group.addButton(rb_circle, 1)
        mode_lay.addWidget(rb_rect)
        mode_lay.addWidget(rb_circle)
        ctrl.addWidget(mode_grp)

        # active rect radio
        rect_grp = QGroupBox("Active crop rect (TL then BR)")
        rect_lay = QVBoxLayout(rect_grp)
        self._rect_btn_group = QButtonGroup(self)
        for i, (name, _, _, color, label) in enumerate(RECTS):
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color: {color.name()};")
            rb.toggled.connect(lambda checked, n=name: checked and self.canvas.set_active_rect(n))
            self._rect_btn_group.addButton(rb, i)
            rect_lay.addWidget(rb)
        self._rect_btn_group.button(0).setChecked(True)
        ctrl.addWidget(rect_grp)

        # manual circle fit
        circ_grp = QGroupBox("Manual circle fit (3 points)")
        circ_lay = QGridLayout(circ_grp)
        self._circle_slot_group = QButtonGroup(self)
        self._circle_pt_labels = []
        for i in range(3):
            rb = QRadioButton(f"Point {i+1}")
            rb.toggled.connect(lambda checked, idx=i: checked and self.canvas.set_active_circle_slot(idx))
            self._circle_slot_group.addButton(rb, i)
            lbl = QLabel("—")
            lbl.setStyleSheet("color: gray; font-size: 10px;")
            circ_lay.addWidget(rb,  i, 0)
            circ_lay.addWidget(lbl, i, 1)
            self._circle_pt_labels.append(lbl)
        self._circle_slot_group.button(0).setChecked(True)

        self._circle_result_label = QLabel("Fit: —")
        self._circle_result_label.setStyleSheet("font-size: 10px;")
        circ_lay.addWidget(self._circle_result_label, 3, 0, 1, 2)

        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self._compute_circle)
        circ_lay.addWidget(calc_btn, 4, 0)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_circle)
        circ_lay.addWidget(clear_btn, 4, 1)
        ctrl.addWidget(circ_grp)

        # threshold sliders
        thresh_grp = QGroupBox("Thresholds")
        thresh_lay = QGridLayout(thresh_grp)
        thresh_lay.addWidget(QLabel("Laser thresh"), 0, 0)
        self.laser_thresh_sl = self._make_slider(0, 255, self._on_thresh_changed)
        thresh_lay.addWidget(self.laser_thresh_sl, 0, 1)
        self.laser_thresh_val = QLabel()
        thresh_lay.addWidget(self.laser_thresh_val, 0, 2)

        thresh_lay.addWidget(QLabel("Leaflet thresh"), 1, 0)
        self.leafl_thresh_sl = self._make_slider(0, 255, self._on_thresh_changed)
        thresh_lay.addWidget(self.leafl_thresh_sl, 1, 1)
        self.leafl_thresh_val = QLabel()
        thresh_lay.addWidget(self.leafl_thresh_val, 1, 2)

        thresh_lay.addWidget(QLabel("Method select thresh"), 2, 0)
        self.method_thresh_sl = self._make_slider(0, 10000, self._on_thresh_changed)
        thresh_lay.addWidget(self.method_thresh_sl, 2, 1)
        self.method_thresh_val = QLabel()
        thresh_lay.addWidget(self.method_thresh_val, 2, 2)

        self.wpc_label = QLabel("WPC: —")
        thresh_lay.addWidget(self.wpc_label, 3, 0, 1, 3)
        ctrl.addWidget(thresh_grp)

        # mask toggle
        self.mask_combo = QComboBox()
        self.mask_combo.addItems(["No mask", "Laser mask", "Leaflet mask"])
        self.mask_combo.currentIndexChanged.connect(self._on_thresh_changed)
        ctrl.addWidget(self.mask_combo)

        ctrl.addStretch()
        return w

    def _build_tab2(self):
        w = QWidget()
        lay = QGridLayout(w)
        row = 0

        def add_range(label, attr_start, attr_end):
            nonlocal row
            lay.addWidget(QLabel(label), row, 0)
            sb_s = QSpinBox(); sb_s.setRange(0, 99999)
            sb_e = QSpinBox(); sb_e.setRange(0, 99999)
            sb_s.valueChanged.connect(lambda v, a=attr_start: self._set_cfg(a, v))
            sb_e.valueChanged.connect(lambda v, a=attr_end:   self._set_cfg(a, v))
            lay.addWidget(sb_s, row, 1)
            lay.addWidget(QLabel("—"), row, 2)
            lay.addWidget(sb_e, row, 3)
            row += 1
            return sb_s, sb_e

        def add_int(label, attr, lo, hi):
            nonlocal row
            lay.addWidget(QLabel(label), row, 0)
            sb = QSpinBox(); sb.setRange(lo, hi)
            sb.valueChanged.connect(lambda v, a=attr: self._set_cfg(a, v))
            lay.addWidget(sb, row, 1, 1, 3)
            row += 1
            return sb

        def add_float(label, attr, lo, hi, step):
            nonlocal row
            lay.addWidget(QLabel(label), row, 0)
            sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setSingleStep(step); sb.setDecimals(3)
            sb.valueChanged.connect(lambda v, a=attr: self._set_cfg(a, v))
            lay.addWidget(sb, row, 1, 1, 3)
            row += 1
            return sb

        self.frame_range_sb    = add_range("Frame range",        "_frame_range_start", "_frame_range_end")
        self.calib_range_sb    = add_range("Calib frame range",  "_calib_start",       "_calib_end")
        self.strip_width_sb    = add_int("Laser strip width",    "laser_strip_width",  1, 100)
        self.peak_step_sb      = add_int("Laser peak step",      "laser_peak_step",    1, 100)
        self.min_peak_sb       = add_int("Laser min peak",       "laser_min_peak",     0, 255)
        self.smooth_cutoff_sb  = add_float("Smooth cutoff",      "smooth_cutoff",      0.01, 1.0, 0.01)
        lay.setRowStretch(row, 1)
        return w

    @staticmethod
    def _make_slider(lo, hi, slot):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.valueChanged.connect(slot)
        return sl

    # ── dataset / frame loading ───────────────────────────────────────────────
    def _populate_datasets(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        subdirs = [d for d in sorted(os.listdir(self.data_dir))
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        for d in subdirs if subdirs else ["."]:
            self.dataset_combo.addItem(d)
        self.dataset_combo.blockSignals(False)
        self._on_dataset_changed(self.dataset_combo.currentText())

    def _on_dataset_changed(self, name):
        folder = self.data_dir if name == "." else os.path.join(self.data_dir, name)
        self._dataset_folder = folder
        self._dataset_name   = name if name != "." else os.path.basename(os.path.abspath(self.data_dir))

        def _numeric_frames(pattern):
            return sorted(
                f for f in glob.glob(pattern)
                if f.split("_")[-1].split(".")[0].isdigit()
            )

        frames = _numeric_frames(f"{folder}/Set_01_*.png")
        if not frames:
            frames = _numeric_frames(f"{self.data_dir}/Set_01_*.png")
            self._dataset_folder = self.data_dir
        self._frames = frames

        nums = [int(f.split("_")[-1].split(".")[0]) for f in frames]
        if nums:
            self.frame_spin.setRange(min(nums), max(nums))
            self.frame_spin.setValue(nums[0])

        calib = os.path.join(folder, "config_custom.json")
        if not os.path.exists(calib):
            calib = "config_default.json"
        self.cfg = load_config(calib)
        self.canvas.set_cfg(self.cfg)
        self._clear_circle()
        self._populate_widgets()
        self._load_frame(self.frame_spin.value())

    def _load_frame(self, num):
        path = next((f for f in self._frames
                     if int(f.split("_")[-1].split(".")[0]) == num), None)
        if path is None and self._frames:
            path = self._frames[0]
        if path is None:
            return
        self._image = cv2.imread(path)
        self.canvas.set_base_image(self._image)
        self._update_wpc()
        self._on_thresh_changed()

    # ── widget ↔ cfg sync ─────────────────────────────────────────────────────
    def _populate_widgets(self):
        c = self.cfg
        self.laser_thresh_sl.setValue(c.get("img_laser_thresh", 90))
        self.leafl_thresh_sl.setValue(c.get("img_leafl_thresh", 20))
        self.method_thresh_sl.setValue(c.get("method_select_thresh", 400))

        fr = c.get("frame_range", [0, 2000])
        cr = c.get("leaflet_calib_frame_range", [0, 200])
        self.frame_range_sb[0].setValue(fr[0])
        self.frame_range_sb[1].setValue(fr[1])
        self.calib_range_sb[0].setValue(cr[0])
        self.calib_range_sb[1].setValue(cr[1])
        self.strip_width_sb.setValue(c.get("laser_strip_width", 10))
        self.peak_step_sb.setValue(c.get("laser_peak_step", 10))
        self.min_peak_sb.setValue(c.get("laser_min_peak", 100))
        self.smooth_cutoff_sb.setValue(c.get("smooth_cutoff", 0.1))

    def _set_cfg(self, key, value):
        mapping = {
            "_frame_range_start": ("frame_range", 0),
            "_frame_range_end":   ("frame_range", 1),
            "_calib_start":       ("leaflet_calib_frame_range", 0),
            "_calib_end":         ("leaflet_calib_frame_range", 1),
        }
        if key in mapping:
            cfg_key, idx = mapping[key]
            lst = list(self.cfg.get(cfg_key, [0, 0]))
            lst[idx] = value
            self.cfg[cfg_key] = lst
        else:
            self.cfg[key] = value

    def _on_rect_changed(self):
        self._on_thresh_changed()

    def _on_thresh_changed(self):
        self.cfg["img_laser_thresh"]     = self.laser_thresh_sl.value()
        self.cfg["img_leafl_thresh"]     = self.leafl_thresh_sl.value()
        self.cfg["method_select_thresh"] = self.method_thresh_sl.value()
        self.laser_thresh_val.setText(str(self.laser_thresh_sl.value()))
        self.leafl_thresh_val.setText(str(self.leafl_thresh_sl.value()))
        self.method_thresh_val.setText(str(self.method_thresh_sl.value()))
        self._update_wpc()
        self._update_mask()

    def _update_wpc(self):
        if self._image is None:
            return
        tl = self.cfg.get("img_leafl_tl", [0, 0])
        br = self.cfg.get("img_leafl_br", [100, 100])
        crop = self._image[tl[1]:br[1], tl[0]:br[0]]
        if crop.size == 0:
            return
        binary = preprocess_image(crop, self.cfg.get("img_leafl_thresh", 20))
        wpc    = count_white_pixels(binary)
        limit  = self.cfg.get("method_select_thresh", 400)
        self.wpc_label.setText(f"WPC: {wpc}  →  {'method2_leaflet' if wpc >= limit else 'method1_laser'}")

    def _update_mask(self):
        if self._image is None:
            self.canvas.clear_mask()
            return
        mode = self.mask_combo.currentIndex()
        if mode == 0:
            self.canvas.clear_mask()
            return
        if mode == 1:
            tl     = self.cfg.get("img_laser_tl",   [0, 0])
            br     = self.cfg.get("img_laser_br",    [100, 100])
            thresh = self.cfg.get("img_laser_thresh", 90)
        else:
            tl     = self.cfg.get("img_leafl_tl",   [0, 0])
            br     = self.cfg.get("img_leafl_br",    [100, 100])
            thresh = self.cfg.get("img_leafl_thresh", 20)
        crop = self._image[tl[1]:br[1], tl[0]:br[0]]
        if crop.size == 0:
            self.canvas.clear_mask()
            return
        binary  = preprocess_image(crop, thresh)
        overlay = np.zeros_like(self._image)
        overlay[tl[1]:br[1], tl[0]:br[0]] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.canvas.set_mask_overlay(overlay)

    # ── circle fit ────────────────────────────────────────────────────────────
    def _on_circle_clicked(self, x, y, slot):
        frame_num = self.frame_spin.value()
        self._circle_frame_nums[slot] = frame_num
        self._circle_pt_labels[slot].setText(f"F{frame_num}  ({x}, {y})")
        # advance to next slot
        next_slot = (slot + 1) % 3
        self._circle_slot_group.button(next_slot).setChecked(True)

    def _compute_circle(self):
        pts = self.canvas._circle_pts
        if any(p is None for p in pts):
            return
        # convert full-image coords to leaflet crop space
        tl = self.cfg.get("img_leafl_tl", [0, 0])
        crop_pts = np.array([[p[0] - tl[0], p[1] - tl[1]] for p in pts], dtype=np.float64)
        cx_crop, cy_crop, r = fit_circle(crop_pts)
        if cx_crop is None:
            self._circle_result_label.setText("Fit failed")
            return
        # convert back to full-image coords for display
        cx = cx_crop + tl[0]
        cy = cy_crop + tl[1]
        self._circle_result_label.setText(f"cx={cx:.1f}  cy={cy:.1f}  r={r:.1f}")
        self.canvas.set_circle_result(cx, cy, r)

    def _clear_circle(self):
        self._circle_frame_nums = [None, None, None]
        for lbl in self._circle_pt_labels:
            lbl.setText("—")
        self._circle_result_label.setText("Fit: —")
        self.canvas.clear_circle_pts()

    # ── save / restore ───────────────────────────────────────────────────────
    def _save(self):
        out_path = os.path.join(self._dataset_folder, "config_custom.json")
        save_config(self.cfg, out_path)
        self.statusBar().showMessage(f"Saved: {out_path}", 4000)

    def _restore_default(self):
        self.cfg = load_config("config_default.json")
        self.canvas.set_cfg(self.cfg)
        self._clear_circle()
        self._populate_widgets()
        self.statusBar().showMessage("Restored default config (not saved)", 4000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = ConfigWindow(data_dir=args.data)
    win.resize(1100, 750)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

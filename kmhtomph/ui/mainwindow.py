"""
Fenêtre principale : ouverture vidéo, lecture, OCR temps réel, export,
barre de progression + raccourcis, extraction ROI par homographie exacte.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, QSettings
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QAction, QFileDialog, QMessageBox, QApplication,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QProgressDialog, QSlider
)

from pytesseract import TesseractNotFoundError

from ..constants import (
    KMH_TO_MPH,
    DEFAULT_SHOW_DEBUG_THUMB,
    DEFAULT_DEBUG_THUMB_SIZE,
)
from ..ocr import OCRPipeline
from ..ocr.tesseract import auto_locate_tesseract
from ..video.io import VideoReader
from ..video.exporter import export_video, ExportParams
from ..video.overlay import OverlayStyle, draw_speed_overlay, format_speed_text
from .canvas import VideoCanvas
from .settings import SettingsDialog


def _extract_roi_from_corners(
    frame_bgr: np.ndarray,
    corners_xy: np.ndarray,  # shape (4,2) float32 : TL,TR,BR,BL en coordonnées image
    w: int,
    h: int,
) -> np.ndarray:
    """Extrait exactement la zone définie par `corners_xy` vers un patch (h, w)."""
    w, h = int(w), int(h)
    if w <= 1 or h <= 1:
        return np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)

    src_pts = np.asarray(corners_xy, dtype=np.float32)
    dst_pts = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float32,
    )
    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    patch = cv2.warpPerspective(
        frame_bgr, Hmat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("kmh→mph OCR")
        self.setFocusPolicy(Qt.StrongFocus)  # capter les raccourcis

        # --- état ---
        self.reader: Optional[VideoReader] = None
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.ocr = OCRPipeline()
        self.ocr_mode = "auto"  # "auto" | "sevenseg" | "tesseract"

        self.overlay_style = OverlayStyle()
        self.show_debug_thumb = DEFAULT_SHOW_DEBUG_THUMB
        self.debug_thumb_size = DEFAULT_DEBUG_THUMB_SIZE

        self._settings = QSettings("kmhtomph", "kmh_to_mph")
        self._tesseract_error_shown = False

        self._tesseract_path: Optional[str] = None
        stored_path = self._settings.value("tesseract/path", type=str)
        if stored_path:
            self._tesseract_path = stored_path
        self._apply_tesseract_path(initial=True)

        # --- UI ---
        self.canvas = VideoCanvas(self)

        self.lbl_kmh = QLabel("-- km/h", self)
        self.lbl_mph = QLabel("-- mph", self)
        for l in (self.lbl_kmh, self.lbl_mph):
            l.setStyleSheet("font-size: 18px;")

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["auto", "sevenseg", "tesseract"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.btn_open = QPushButton("Ouvrir…", self)
        self.btn_play = QPushButton("Lecture", self)
        self.btn_export = QPushButton("Exporter…", self)

        self.chk_debug = QCheckBox("Vignette debug", self)
        self.chk_debug.setChecked(self.show_debug_thumb)
        self.chk_debug.stateChanged.connect(self._on_toggle_debug)

        self.spin_dbg = QSpinBox(self)
        self.spin_dbg.setRange(64, 512)
        self.spin_dbg.setValue(self.debug_thumb_size)
        self.spin_dbg.setSuffix(" px")
        self.spin_dbg.valueChanged.connect(self._on_thumb_size)

        # --- Barre de progression + temps ---
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(25)

        self.lbl_time = QLabel("00:00 / 00:00", self)

        self._seeking = False
        self._was_playing_before_seek = False

        # layout
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_play)
        top_bar.addWidget(self.btn_export)
        top_bar.addSpacing(20)
        top_bar.addWidget(QLabel("Mode OCR:", self))
        top_bar.addWidget(self.mode_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.lbl_kmh)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.lbl_mph)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chk_debug)
        top_bar.addWidget(self.spin_dbg)

        prog_bar = QHBoxLayout()
        prog_bar.addWidget(self.slider, 1)
        prog_bar.addSpacing(8)
        prog_bar.addWidget(self.lbl_time)

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.addLayout(top_bar)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(prog_bar)
        self.setCentralWidget(central)

        # actions
        self.btn_open.clicked.connect(self._on_open)
        self.btn_play.clicked.connect(self._on_toggle_play)
        self.btn_export.clicked.connect(self._on_export)

        # slider signals
        self.slider.sliderPressed.connect(self._on_seek_start)
        self.slider.sliderReleased.connect(self._on_seek_end)
        self.slider.valueChanged.connect(self._on_seek_change)

        # menu
        self._create_menu()

        self.resize(1000, 740)

    # ------------- Menu -------------

    def _create_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Fichier")
        act_open = QAction("Ouvrir…", self)
        act_export = QAction("Exporter…", self)
        act_quit = QAction("Quitter", self)
        act_open.triggered.connect(self._on_open)
        act_export.triggered.connect(self._on_export)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_open)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        edit_menu = menubar.addMenu("&Édition")
        act_settings = QAction("Paramètres OCR…", self)
        act_settings.triggered.connect(self._on_settings)
        edit_menu.addAction(act_settings)

        view_menu = menubar.addMenu("&Affichage")
        act_fit_roi = QAction("Ajuster le ROI à l’image", self)
        act_fit_roi.triggered.connect(self.canvas.fit_roi_to_frame)
        view_menu.addAction(act_fit_roi)

    # ------------- Utilitaires -------------

    def _format_hms(self, seconds: float) -> str:
        if seconds < 0 or not np.isfinite(seconds):
            return "00:00"
        s = int(round(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _fps(self) -> float:
        return float(self.reader.fps or 25.0) if self.reader else 25.0

    def _total_frames(self) -> int:
        return int(self.reader.frame_count) if (self.reader and self.reader.frame_count > 0) else 0

    # ------------- Raccourcis clavier -------------

    def keyPressEvent(self, ev) -> None:
        if not self.reader or not self.reader.is_opened():
            return super().keyPressEvent(ev)
        key = ev.key()

        if key == Qt.Key_Space:
            self._on_toggle_play(); ev.accept(); return
        if key == Qt.Key_Left:
            self._step_frames(-1); ev.accept(); return
        if key == Qt.Key_Right:
            self._step_frames(+1); ev.accept(); return
        if key == Qt.Key_PageUp:
            self._step_seconds(+1.0); ev.accept(); return
        if key == Qt.Key_PageDown:
            self._step_seconds(-1.0); ev.accept(); return
        if key == Qt.Key_Home:
            self._seek_to_frame(0); ev.accept(); return
        if key == Qt.Key_End:
            self._seek_to_frame(max(0, self._total_frames() - 1)); ev.accept(); return

        super().keyPressEvent(ev)

    def _step_frames(self, delta_frames: int) -> None:
        if not self.reader:
            return
        was_playing = self.playing
        if was_playing:
            self._on_toggle_play()  # pause
        cur_ms = self.reader.get_pos_msec()
        cur = int(round((cur_ms / 1000.0) * self._fps()))
        target = int(np.clip(cur + delta_frames, 0, max(0, self._total_frames() - 1)))
        self._seek_to_frame(target)
        if was_playing:
            self._on_toggle_play()

    def _step_seconds(self, delta_seconds: float) -> None:
        frames = int(round(delta_seconds * self._fps()))
        self._step_frames(frames)

    def _seek_to_frame(self, target_frame: int) -> None:
        if not self.reader:
            return
        self._seeking = True
        try:
            if not self.reader.set_pos_frame(target_frame):
                ms = (target_frame / self._fps()) * 1000.0
                self.reader.seek_msec(ms)
            ok, frame = self.reader.read()
            if ok and frame is not None:
                self.canvas.set_frame(frame)

            if self.slider.isEnabled():
                self.slider.blockSignals(True)
                self.slider.setValue(max(0, min(target_frame, max(0, self._total_frames() - 1))))
                self.slider.blockSignals(False)

            total_seconds = (self._total_frames() / self._fps()) if self._total_frames() > 0 else 0.0
            self.lbl_time.setText(f"{self._format_hms(target_frame / self._fps())} / {self._format_hms(total_seconds)}")
        finally:
            self._seeking = False

    # ------------- Slots -------------

    def _on_mode_changed(self, txt: str):
        self.ocr_mode = txt
        self.ocr.reset()

    def _on_toggle_debug(self, state: int):
        self.show_debug_thumb = (state == Qt.Checked)

    def _on_thumb_size(self, v: int):
        self.debug_thumb_size = int(v)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une vidéo", "", "Vidéos (*.mp4 *.m4v *.mov *.avi *.mkv);;Tous les fichiers (*)")
        if not path:
            return
        try:
            if self.reader is not None:
                self.reader.release()
            self.reader = VideoReader(path)
            self.reader.open()
        except Exception as e:
            QMessageBox.critical(self, "Ouverture", f"Échec : {e}")
            self.reader = None
            return

        ok, frame = self.reader.read()
        if not ok or frame is None:
            QMessageBox.warning(self, "Ouverture", "Impossible de lire la vidéo.")
            self.reader.release()
            self.reader = None
            return

        total_frames = self._total_frames()
        fps = self._fps()
        self.slider.setEnabled(total_frames > 0)
        self.slider.setRange(0, max(0, total_frames - 1))
        self.slider.setValue(0)
        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        self.lbl_time.setText(f"00:00 / {self._format_hms(total_seconds)}")

        self.canvas.set_frame(frame)
        self.canvas.fit_roi_to_frame()
        self.lbl_kmh.setText("-- km/h")
        self.lbl_mph.setText("-- mph")
        self.ocr.reset()

    def _on_toggle_play(self):
        if not self.reader or not self.reader.is_opened():
            return
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Lecture")
        if self.playing:
            self.timer.start(int(1000 / max(1.0, self._fps())))
        else:
            self.timer.stop()

    def _on_tick(self):
        if not self.reader:
            return
        ok, frame = self.reader.read()
        if not ok or frame is None:
            self._on_toggle_play()
            return

        self.canvas.set_frame(frame)

        # OCR sur le ROI courant — coins exacts du canvas
        cx, cy, w, h, ang = self.canvas.get_roi()
        corners = self.canvas.get_roi_corners()  # (4,2) TL,TR,BR,BL en coords image
        roi = _extract_roi_from_corners(frame, corners, w, h)

        try:
            kmh, debug_bgr, score, details = self.ocr.read_kmh(roi, mode=self.ocr_mode)
        except (TesseractNotFoundError, FileNotFoundError) as e:
            self._handle_tesseract_error(e)
            return

        if self.show_debug_thumb:
            if debug_bgr is not None and debug_bgr.size > 0:
                thumb = cv2.resize(debug_bgr, (self.debug_thumb_size, self.debug_thumb_size), interpolation=cv2.INTER_NEAREST)
                self.canvas.set_debug_thumb(thumb)
            else:
                self.canvas.clear_debug_thumb()
        else:
            self.canvas.clear_debug_thumb()

        if kmh is not None:
            mph = kmh * KMH_TO_MPH
            self.lbl_kmh.setText(f"{kmh:.0f} km/h")
            self.lbl_mph.setText(format_speed_text(mph))
        else:
            self.lbl_kmh.setText("-- km/h")
            self.lbl_mph.setText("-- mph")

        # --- progression ---
        cur_ms = self.reader.get_pos_msec()
        fps = self._fps()
        cur_frame = int(round((cur_ms / 1000.0) * fps)) if fps > 0 else 0
        total_frames = self._total_frames()

        if not self._seeking and self.slider.isEnabled():
            self.slider.blockSignals(True)
            self.slider.setValue(max(0, min(cur_frame, max(0, total_frames - 1))))
            self.slider.blockSignals(False)

        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        self.lbl_time.setText(f"{self._format_hms(cur_ms/1000.0)} / {self._format_hms(total_seconds)}")

    def _on_settings(self):
        dlg = SettingsDialog(self, initial_path=self._tesseract_path)
        if dlg.exec_():
            prev_path = self._tesseract_path
            self._tesseract_path = dlg.tesseract_path or None
            if not self._apply_tesseract_path():
                self._tesseract_path = prev_path
                self._apply_tesseract_path(initial=True)

    def _apply_tesseract_path(self, *, initial: bool = False) -> bool:
        path = self._tesseract_path or None
        try:
            auto_locate_tesseract(path)
        except FileNotFoundError as e:
            if not initial:
                QMessageBox.warning(
                    self,
                    "Tesseract introuvable",
                    "Impossible de trouver l'exécutable Tesseract.\n"
                    "Vérifiez l'installation ou choisissez le bon fichier dans les Paramètres OCR.\n"
                    f"Détail : {e}",
                )
            return False
        else:
            self._tesseract_error_shown = False
            if path:
                self._save_tesseract_path()
            elif not initial:
                self._settings.remove("tesseract/path")
                self._settings.sync()
            return True

    def _save_tesseract_path(self) -> None:
        if self._tesseract_path:
            self._settings.setValue("tesseract/path", self._tesseract_path)
        else:
            self._settings.remove("tesseract/path")
        self._settings.sync()

    def _handle_tesseract_error(self, err: Exception) -> None:
        if self._tesseract_error_shown:
            return
        self._tesseract_error_shown = True
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.btn_play.setText("Lecture")
        self.canvas.clear_debug_thumb()
        self.lbl_kmh.setText("-- km/h")
        self.lbl_mph.setText("-- mph")
        QMessageBox.critical(
            self,
            "Erreur Tesseract",
            "Tesseract n'a pas pu être exécuté.\n"
            "Veuillez vérifier l'installation et configurer le chemin dans Paramètres OCR…\n"
            f"Détail : {err}",
        )

    # ------------- Export -------------

    def _on_export(self):
        if not self.reader or not self.reader.is_opened():
            QMessageBox.information(self, "Export", "Ouvrez d’abord une vidéo.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Exporter la vidéo", "", "MP4 (*.mp4);;AVI (*.avi);;Tous les fichiers (*)")
        if not out_path:
            return

        src = self.reader.source
        new_reader = VideoReader(src)
        try:
            new_reader.open()
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Impossible de rouvrir la source : {e}")
            return

        # Capturer le ROI & coins actuels
        cx, cy, w, h, ang = self.canvas.get_roi()
        base_corners = self.canvas.get_roi_corners().copy()
        style = self.overlay_style

        last_mph_value = {"value": None}

        def text_supplier(idx: int, frame_bgr: np.ndarray) -> Optional[str]:
            # Recalcule une fois (si on souhaite geler la position au début) : ici on garde base_corners
            roi_bgr = _extract_roi_from_corners(frame_bgr, base_corners, w, h)
            try:
                kmh, debug_bgr, score, details = self.ocr.read_kmh(roi_bgr, mode=self.ocr_mode)
            except (TesseractNotFoundError, FileNotFoundError) as e:
                self._handle_tesseract_error(e)
                raise RuntimeError("Tesseract introuvable") from e
            if kmh is None:
                return None
            mph = kmh * KMH_TO_MPH
            last_mph_value["value"] = mph
            return format_speed_text(mph)

        def draw_overlay(frame_bgr: np.ndarray, text: str) -> None:
            value = last_mph_value["value"]
            if value is None:
                try:
                    value = float(text.split()[0])
                except (ValueError, IndexError):
                    return
            draw_speed_overlay(frame_bgr, value, (cx, cy), ang, style, text=text)

        total = int(new_reader.frame_count) if new_reader.frame_count > 0 else None
        progress = QProgressDialog("Export en cours…", "Annuler", 0, total or 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        canceled = {"flag": False}

        def on_progress(done: int, total_opt: Optional[int]):
            if total_opt:
                progress.setMaximum(total_opt)
                progress.setValue(done)
            else:
                progress.setValue(done % 100)
            QApplication.processEvents()
            if progress.wasCanceled():
                canceled["flag"] = True
                raise RuntimeError("Export annulé")

        try:
            export_video(
                new_reader,
                out_path,
                text_supplier=text_supplier,
                draw_overlay=draw_overlay,
                on_progress=on_progress,
                params=ExportParams(),
            )
            progress.setValue(progress.maximum())
            QMessageBox.information(self, "Export", f"Fichier écrit :\n{out_path}")
        except Exception as e:
            if not canceled["flag"]:
                QMessageBox.warning(self, "Export", f"Échec : {e}")
        finally:
            try:
                new_reader.release()
            except Exception:
                pass

    # ------------- Seek / Slider -------------

    def _on_seek_start(self):
        if not self.reader or not self.slider.isEnabled():
            return
        self._seeking = True
        self._was_playing_before_seek = self.playing
        if self.playing:
            self._on_toggle_play()  # pause

    def _on_seek_change(self, value: int):
        if not self.reader:
            return
        fps = self._fps()
        total_frames = self._total_frames()
        total_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0
        cur_seconds = (value / fps) if fps > 0 else 0.0
        self.lbl_time.setText(f"{self._format_hms(cur_seconds)} / {self._format_hms(total_seconds)}")

    def _on_seek_end(self):
        if not self.reader or not self.slider.isEnabled():
            self._seeking = False
            return
        target_frame = int(self.slider.value())
        try:
            if not self.reader.set_pos_frame(target_frame):
                ms = (target_frame / self._fps()) * 1000.0
                self.reader.seek_msec(ms)
            ok, frame = self.reader.read()
            if ok and frame is not None:
                self.canvas.set_frame(frame)
            else:
                self.reader.set_pos_frame(max(0, target_frame - 1))
                ok, frame = self.reader.read()
                if ok and frame is not None:
                    self.canvas.set_frame(frame)
        finally:
            self._seeking = False
            if self._was_playing_before_seek:
                self._on_toggle_play()  # reprendre

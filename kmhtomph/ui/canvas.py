from __future__ import annotations

from typing import Optional, Tuple
import math

import numpy as np
import cv2

# Import modulaires Qt (meilleur pour les analyseurs & stubs)
from PyQt5 import QtCore, QtGui, QtWidgets


def _bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    """Convertit un BGR OpenCV en QImage RGB888 (copie)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
    return qimg.copy()  # détacher du buffer numpy


class VideoCanvas(QtWidgets.QWidget):
    on_roi_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_bgr: Optional[np.ndarray] = None
        self._debug_bgr: Optional[np.ndarray] = None

        # ROI : centre, taille, angle (deg)
        self._cx = 200
        self._cy = 150
        self._w = 180
        self._h = 90
        self._angle = 0.0

        # Interaction
        self._dragging_move = False
        self._dragging_resize = False
        self._drag_anchor: Tuple[int, int] | None = None
        self._resize_corner = 0  # 0..3

        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # ------------- API publique -------------

    def sizeHint(self) -> QtCore.QSize:
        if self._frame_bgr is not None:
            h, w = self._frame_bgr.shape[:2]
            return QtCore.QSize(w, h)
        return super().sizeHint()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, "frame_bgr doit être BGR HxWx3"
        self._frame_bgr = frame_bgr.copy()
        self.update()

    def set_debug_thumb(self, debug_bgr: Optional[np.ndarray]) -> None:
        self._debug_bgr = debug_bgr.copy() if debug_bgr is not None else None
        self.update()

    def clear_debug_thumb(self) -> None:
        self._debug_bgr = None
        self.update()

    def set_roi(self, cx: int, cy: int, w: int, h: int, angle_deg: float = 0.0) -> None:
        self._cx, self._cy, self._w, self._h, self._angle = int(cx), int(cy), int(w), int(h), float(angle_deg)
        self._normalize_roi()
        self.on_roi_changed.emit()
        self.update()

    def fit_roi_to_frame(self, margin: int = 20) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        self._cx, self._cy = w // 2, h // 2
        self._w = max(10, w - 2 * margin)
        self._h = max(10, h // 5)
        self._angle = 0.0
        self.on_roi_changed.emit()
        self.update()

    def get_roi(self) -> tuple[int, int, int, int, float]:
        return int(self._cx), int(self._cy), int(self._w), int(self._h), float(self._angle)

    def get_roi_corners(self) -> np.ndarray:
        """
        Retourne les 4 coins du ROI tourné en coordonnées image (float32),
        dans l'ordre : top-left, top-right, bottom-right, bottom-left.

        Convention : le rectangle affiché est tourné **horaire** de `self._angle`.
        """
        cx, cy, w, h, angle = self.get_roi()
        half_w, half_h = w / 2.0, h / 2.0

        # Coins locaux (rectangle centré)
        local = np.array(
            [
                [-half_w, -half_h],  # TL
                [ half_w, -half_h],  # TR
                [ half_w,  half_h],  # BR
                [-half_w,  half_h],  # BL
            ],
            dtype=np.float32,
        )

        # Rotation HORAIRE de `angle` en repère image (y vers le bas)
        # R_clock = [[ cos,  sin],
        #            [-sin,  cos]]
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        R_clock = np.array([[c, s], [-s, c]], dtype=np.float32)

        pts = (local @ R_clock.T) + np.array([cx, cy], dtype=np.float32)
        return pts  # (4,2) float32

    # ------------- Dessin -------------

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QtGui.QColor(20, 20, 20))

        # Frame vidéo
        if self._frame_bgr is not None:
            qimg = _bgr_to_qimage(self._frame_bgr)
            p.drawImage(
                0, 0,
                qimg.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation),
            )

        # Transform image -> widget
        sx, sy, ox, oy = self._image_transform()

        # ROI style
        pen = QtGui.QPen(QtGui.QColor(255, 180, 0), 2, QtCore.Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)

        # Coord en widget
        cx = ox + self._cx * sx
        cy = oy + self._cy * sy
        w = self._w * sx
        h = self._h * sy

        # Rotation autour du centre
        t = QtGui.QTransform()
        t.translate(cx, cy)
        t.rotate(-self._angle)  # Qt est antihoraire -> signe -
        p.setTransform(t, combine=False)

        # Rectangle centré (QRectF pour floats)
        x0 = -w / 2.0
        y0 = -h / 2.0
        p.drawRect(QtCore.QRectF(x0, y0, w, h))

        # Poignées (coins)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 180, 0)))
        r = 6.0
        for (px, py) in [(-w/2.0, -h/2.0), (w/2.0, -h/2.0), (w/2.0, h/2.0), (-w/2.0, h/2.0)]:
            p.drawEllipse(QtCore.QPointF(px, py), r, r)

        # Reset transform
        p.resetTransform()

        # Vignette debug
        if self._debug_bgr is not None and self._debug_bgr.size > 0:
            thumb = _bgr_to_qimage(self._debug_bgr)
            tw = min(200, max(1, self.width() // 3))
            th = int(thumb.height() * (tw / max(1, thumb.width())))
            x = self.width() - tw - 10
            y = self.height() - th - 10

            p.setOpacity(0.95)
            p.fillRect(x - 2, y - 2, tw + 4, th + 4, QtGui.QColor(0, 0, 0, 140))
            p.setOpacity(1.0)

            p.drawImage(
                x, y,
                thumb.scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation),
            )

        p.end()

    # ------------- Interaction souris -------------

    def mousePressEvent(self, ev) -> None:
        if self._frame_bgr is None:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            self._dragging_move = True
            self._drag_anchor = (ev.x(), ev.y())
        elif ev.button() == QtCore.Qt.RightButton:
            self._dragging_resize = True
            self._drag_anchor = (ev.x(), ev.y())
            self._resize_corner = self._closest_corner(ev.pos())
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev) -> None:
        if self._frame_bgr is None or self._drag_anchor is None:
            return

        dx = ev.x() - self._drag_anchor[0]
        dy = ev.y() - self._drag_anchor[1]

        sx, sy, ox, oy = self._image_transform()
        ddx = dx / max(1e-6, sx)
        ddy = dy / max(1e-6, sy)

        # rotation inverse pour delta local
        rad = math.radians(self._angle)
        cos, sin = math.cos(rad), math.sin(rad)
        local_dx = cos * ddx + sin * ddy
        local_dy = -sin * ddx + cos * ddy

        if self._dragging_move:
            self._cx += local_dx
            self._cy += local_dy
        elif self._dragging_resize:
            kx = 1 if self._resize_corner in (1, 2) else -1
            ky = 1 if self._resize_corner in (2, 3) else -1
            self._w += kx * 2 * local_dx
            self._h += ky * 2 * local_dy

        self._drag_anchor = (ev.x(), ev.y())
        self._normalize_roi()
        self.on_roi_changed.emit()
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging_move = False
        self._dragging_resize = False
        self._drag_anchor = None
        self.setCursor(QtCore.Qt.ArrowCursor)

    def wheelEvent(self, ev) -> None:
        step = 1.0 if (ev.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier)) else 3.0
        delta = ev.angleDelta().y() / 120.0
        self._angle = (self._angle + delta * step) % 360.0
        self.on_roi_changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, ev) -> None:
        sx, sy, ox, oy = self._image_transform()
        ix = (ev.x() - ox) / max(1e-6, sx)
        iy = (ev.y() - oy) / max(1e-6, sy)
        self._cx, self._cy = ix, iy
        self._normalize_roi()
        self.on_roi_changed.emit()
        self.update()

    # ------------- Helpers internes -------------

    def _image_transform(self) -> Tuple[float, float, float, float]:
        """Retourne (sx, sy, ox, oy) pour le mapping image->widget."""
        if self._frame_bgr is None:
            return 1.0, 1.0, 0.0, 0.0
        h, w = self._frame_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return 1.0, 1.0, 0.0, 0.0
        rw = self.width() / w
        rh = self.height() / h
        s = min(rw, rh)
        sx = sy = s
        ox = (self.width() - w * s) / 2.0
        oy = (self.height() - h * s) / 2.0
        return sx, sy, ox, oy

    def _normalize_roi(self) -> None:
        if self._frame_bgr is None:
            return
        h, w = self._frame_bgr.shape[:2]
        self._w = max(10, min(self._w, w))
        self._h = max(10, min(self._h, h))
        self._cx = float(max(0, min(self._cx, w)))
        self._cy = float(max(0, min(self._cy, h)))

    def _closest_corner(self, pos) -> int:
        """Renvoie l'index du coin le plus proche dans l'espace écran (0..3)."""
        sx, sy, ox, oy = self._image_transform()
        cx = ox + self._cx * sx
        cy = oy + self._cy * sy
        w = self._w * sx
        h = self._h * sy
        rad = math.radians(self._angle)
        cos, sin = math.cos(rad), math.sin(rad)
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        pts = []
        for (x, y) in corners:
            rx = cos * x - sin * y
            ry = sin * x + cos * y
            pts.append((cx + rx, cy + ry))
        px, py = pos.x(), pos.y()
        dists = [(i, (px - x)**2 + (py - y)**2) for i, (x, y) in enumerate(pts)]
        dists.sort(key=lambda t: t[1])
        return dists[0][0]

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, List

import re
import os
import shutil
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass(frozen=True)
class TesseractParams:
    # Prétraitements
    denoise_bilateral: bool = True
    clahe: bool = True
    unsharp: bool = True

    # Mise à l’échelle
    scale_to_height: int = 120

    # Tesseract
    psm: int = 7  # single line
    oem: int = 3  # default LSTM
    allow_dot: bool = False  # pour "12.3"
    whitelist: str = "0123456789"

    # Morphologie
    morph_open: bool = True
    morph_close: bool = True


DEFAULT_PARAMS = TesseractParams()


def _is_executable_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def auto_locate_tesseract(explicit_path: Optional[str] = None) -> str:
    """Configure pytesseract pour utiliser un binaire valide.

    Si ``explicit_path`` est fourni, il est testé en priorité. Sinon, on tente :
    - la valeur déjà configurée dans ``pytesseract.pytesseract.tesseract_cmd`` ;
    - ``shutil.which('tesseract')`` ;
    - quelques emplacements connus sous Windows.

    Retourne le chemin retenu ou lève ``FileNotFoundError`` si aucun binaire n’est
    disponible.
    """

    candidates: List[str] = []

    if explicit_path:
        candidates.append(explicit_path)

    current_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if current_cmd:
        candidates.append(current_cmd)

    found_on_path = shutil.which("tesseract")
    if found_on_path:
        candidates.append(found_on_path)

    if sys.platform.startswith("win"):
        program_files = os.environ.get("PROGRAMFILES", r"C:\\Program Files")
        program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")
        win_defaults = [
            os.path.join(program_files, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(program_files_x86, "Tesseract-OCR", "tesseract.exe"),
        ]
        candidates.extend(win_defaults)

    checked = set()
    for path in candidates:
        if not path or path in checked:
            continue
        checked.add(path)
        if _is_executable_file(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path

    raise FileNotFoundError("Aucun exécutable Tesseract valide trouvé")


def _make_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _unsharp_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp


def _binarize(gray: np.ndarray) -> np.ndarray:
    # OTSU par défaut, inversion auto si besoin
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(thr) < 128:
        thr = cv2.bitwise_not(thr)
    return thr


def _apply_morphology(thr: np.ndarray, p: TesseractParams) -> np.ndarray:
    if p.morph_open:
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    if p.morph_close:
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr


def _prep_for_ocr(gray_in: np.ndarray, p: TesseractParams) -> Tuple[np.ndarray, np.ndarray]:
    g = _make_gray(gray_in)
    # upscale
    h = max(int(p.scale_to_height), 24)
    H, W = g.shape[:2]
    fx = h / max(1, H)
    g = cv2.resize(g, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)

    # filtres
    if p.denoise_bilateral:
        g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=7)
    if p.clahe:
        g = _apply_clahe(g)
    if p.unsharp:
        g = _unsharp_mask(g)

    thr = _apply_morphology(_binarize(g), p)
    return g, thr


def _adaptive_binarize(gray: np.ndarray, p: TesseractParams) -> np.ndarray:
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        2,
    )
    return _apply_morphology(thr, p)


def _looks_mostly_blank(thr: np.ndarray, *, min_fraction: float = 0.03) -> bool:
    if thr.size == 0:
        return True
    white_ratio = float(np.count_nonzero(thr == 255)) / float(thr.size)
    black_ratio = 1.0 - white_ratio
    return white_ratio < min_fraction or black_ratio < min_fraction


def _content_bounds(thr: np.ndarray, *, min_area: int = 64) -> Optional[Tuple[int, int, int, int]]:
    if thr.ndim != 2 or thr.size == 0:
        return None

    unique_vals, counts = np.unique(thr, return_counts=True)
    if len(unique_vals) <= 1:
        return None

    bg_val = int(unique_vals[int(np.argmax(counts))])
    if bg_val not in (0, 255):
        # Fallback : considérer la valeur la plus proche de 0 ou 255 comme fond.
        bg_val = 0 if abs(bg_val - 0) <= abs(bg_val - 255) else 255

    mask = thr != bg_val
    if not np.any(mask):
        return None

    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    if (y1 - y0) * (x1 - x0) < int(min_area):
        return None

    margin_y = max(1, (y1 - y0) // 12)
    margin_x = max(1, (x1 - x0) // 12)

    y0 = max(0, y0 - margin_y)
    y1 = min(thr.shape[0], y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(thr.shape[1], x1 + margin_x)

    return y0, y1, x0, x1


def _crop_to_bounds(img: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    cropped = img[y0:y1, x0:x1]
    return cropped if cropped.size else img


def _run_tesseract(thr: np.ndarray, p: TesseractParams) -> Tuple[Optional[str], float, np.ndarray]:
    data = pytesseract.image_to_data(thr, config=_tess_config(p), output_type=Output.DICT)

    # Extraire texte brut + confiance
    words = data.get("text", [])
    confs = data.get("conf", [])
    txt_raw = "".join(words) if words else ""
    txt_raw = txt_raw.strip()

    # Nettoyage : garder digits + point si autorisé
    if p.allow_dot:
        m = re.findall(r"[0-9]+(?:[.,][0-9])?", txt_raw)
    else:
        m = re.findall(r"[0-9]+", txt_raw)
    txt = "".join(m) if m else ""

    # Confiance moyenne sur les boxes valides
    cvals: List[float] = []
    for c in confs:
        try:
            cv = float(c)
            if cv >= 0:
                cvals.append(cv)
        except Exception:
            pass
    conf = (float(np.mean(cvals)) / 100.0) if cvals else 0.0

    dbg = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    return (txt if txt else None), conf, dbg


def _tess_config(p: TesseractParams) -> str:
    wl = p.whitelist + (".," if p.allow_dot else "")
    config = f"--psm {int(p.psm)} --oem {int(p.oem)} -c tessedit_char_whitelist={wl}"
    return config


def _finalize_and_try(gray_in: np.ndarray, p: TesseractParams) -> Tuple[Optional[str], float, np.ndarray]:
    gray_proc, thr_main = _prep_for_ocr(gray_in, p)

    bounds = _content_bounds(thr_main)
    if bounds:
        gray_proc = _crop_to_bounds(gray_proc, bounds)
        thr_main = _crop_to_bounds(thr_main, bounds)

    variants: List[np.ndarray] = [thr_main, cv2.bitwise_not(thr_main)]

    if _looks_mostly_blank(thr_main):
        thr_adapt = _adaptive_binarize(gray_proc, p)
        if bounds and thr_adapt.shape != thr_main.shape:
            thr_adapt = _crop_to_bounds(thr_adapt, bounds)
        variants.append(thr_adapt)
        variants.append(cv2.bitwise_not(thr_adapt))

    best: Tuple[Optional[str], float, np.ndarray] = (
        None,
        0.0,
        cv2.cvtColor(gray_proc if gray_proc.size else gray_in, cv2.COLOR_GRAY2BGR),
    )
    for thr in variants:
        txt, conf, dbg = _run_tesseract(thr, p)
        if conf >= best[1]:
            best = (txt, conf, dbg)

    return best


def tesseract_ocr(bgr_or_gray: np.ndarray, params: Optional[TesseractParams] = None) -> Tuple[Optional[str], float, np.ndarray]:
    """
    OCR Tesseract avec plusieurs variantes/fallbacks. Retourne (texte, confiance, image_debug).
    Nouveautés :
      - Ajout d’un essai PSM 13 (ligne brute) pour les chiffres serrés
      - Légère hausse d’upscale (jusqu’à 140 px de haut) comme backup
    """
    g = _make_gray(bgr_or_gray)
    p0 = params if params is not None else DEFAULT_PARAMS

    best: Tuple[Optional[str], float, Optional[np.ndarray]] = (None, 0.0, None)
    threshold = 0.85  # si on dépasse, on “early return”

    # Essai principal
    for p in (
        p0,
        replace(p0, psm=(7 if int(p0.psm) != 7 else 8)),             # alterner 7/8
        replace(p0, scale_to_height=max(110, int(p0.scale_to_height))),
        replace(p0, psm=13),                                         # NEW: ligne brute
        replace(p0, scale_to_height=max(140, int(p0.scale_to_height))),  # NEW: plus grand
    ):
        t, c, dbg = _finalize_and_try(g, p)
        if c >= threshold and t is not None:
            return t, c, dbg
        if c > best[1]:
            best = (t, c, dbg)

    # Retour meilleur trouvé
    t, c, dbg = best
    if dbg is None:
        dbg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return (t if t else None), float(c), dbg

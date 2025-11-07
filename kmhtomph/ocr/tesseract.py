from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError


@dataclass(frozen=True)
class TesseractParams:
    """Paramètres de base pour piloter Tesseract.

    Ce jeu réduit d'options colle aux recommandations essentielles de la
    documentation officielle : choisir explicitement le moteur OCR, le mode de
    segmentation de page et les langues, tout en gardant la possibilité de
    borner le jeu de caractères attendu. Le redimensionnement léger reste
    disponible pour améliorer la lisibilité des petites polices.
    """

    lang: str = "eng"
    psm: int = 7
    oem: int = 1
    scale_to_height: int = 120
    allow_decimal: bool = False
    # Autorise les chiffres ainsi que les caractères d'unités fréquents
    # (km/h, mph) pour éviter que Tesseract ne rejette toute la ligne quand
    # l'affichage contient la vitesse suivie de son unité collée.
    whitelist: str = "0123456789kmhKMHmphMPH/ "
    tessdata_dir: Optional[str] = None


DEFAULT_PARAMS = TesseractParams()


def _is_executable_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def auto_locate_tesseract(explicit_path: Optional[str] = None) -> str:
    """Configure ``pytesseract`` pour utiliser un binaire Tesseract disponible.

    L'ordre de détection respecte les recommandations du projet : utiliser le
    chemin explicite, puis la configuration actuelle de ``pytesseract``, puis
    rechercher sur le ``PATH`` et enfin essayer quelques emplacements Windows
    courants.
    """

    candidates: list[str] = []

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

    checked: set[str] = set()
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


def _resize_to_height(gray: np.ndarray, target_height: int) -> np.ndarray:
    if target_height <= 0 or gray.size == 0:
        return gray

    target = max(24, int(target_height))
    h, w = gray.shape[:2]
    if h >= target:
        return gray

    scale = float(target) / float(max(1, h))
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _binarize(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


def _ensure_black_on_white(img: np.ndarray) -> np.ndarray:
    if img.size == 0:
        return img
    mean_val = float(np.mean(img))
    if mean_val < 127.0:
        return cv2.bitwise_not(img)
    return img


def _prepare_image(bgr_or_gray: np.ndarray, params: TesseractParams) -> Tuple[np.ndarray, np.ndarray]:
    gray = _make_gray(bgr_or_gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = _resize_to_height(gray, int(params.scale_to_height))
    binary = _binarize(gray)
    binary = _ensure_black_on_white(binary)
    binary = np.ascontiguousarray(binary.astype(np.uint8))
    padded = cv2.copyMakeBorder(binary, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return gray, padded


def _build_config(params: TesseractParams) -> str:
    whitelist = params.whitelist or ""
    if params.allow_decimal:
        for ch in ".,":
            if ch not in whitelist:
                whitelist += ch

    parts = [
        f"--psm {int(params.psm)}",
        f"--oem {int(params.oem)}",
        "quiet",
        f"-c tessedit_char_whitelist={whitelist}",
        "-c classify_bln_numeric_mode=1",
        "-c load_system_dawg=0",
        "-c load_freq_dawg=0",
    ]

    if params.tessdata_dir:
        parts.append(f"--tessdata-dir \"{params.tessdata_dir}\"")

    return " ".join(parts)


def _extract_numeric_text(data: dict, allow_decimal: bool) -> Optional[str]:
    texts = [t.strip() for t in data.get("text", []) if isinstance(t, str) and t.strip()]
    if not texts:
        return None

    joined = " ".join(texts)
    pattern = r"[0-9]+(?:[.,][0-9]+)?" if allow_decimal else r"[0-9]+"
    matches = re.findall(pattern, joined)
    if not matches:
        return None

    best = max(matches, key=len)
    return best.replace(",", ".") if allow_decimal else best


def _average_confidence(data: dict) -> float:
    confs = []
    for c in data.get("conf", []):
        try:
            val = float(c)
        except (TypeError, ValueError):
            continue
        if val >= 0:
            confs.append(val)
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs)) / 100.0


def _render_debug(prepared: np.ndarray, data: dict, text: Optional[str], conf: float) -> np.ndarray:
    if prepared.ndim != 2:
        prepared = _make_gray(prepared)
    dbg = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    n = len(data.get("level", []))
    for idx in range(n):
        try:
            conf_val = float(data["conf"][idx])
        except (KeyError, TypeError, ValueError):
            conf_val = -1.0
        if conf_val < 0:
            continue

        try:
            x = int(data["left"][idx])
            y = int(data["top"][idx])
            w = int(data["width"][idx])
            h = int(data["height"][idx])
        except (KeyError, TypeError, ValueError):
            continue

        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 200, 0), 1)

    label = text if text else "--"
    baseline_y = max(16, dbg.shape[0] - 8)
    cv2.putText(
        dbg,
        f"{label} ({conf:.2f})",
        (6, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return dbg


def tesseract_ocr(
    bgr_or_gray: np.ndarray, params: Optional[TesseractParams] = None
) -> Tuple[Optional[str], float, np.ndarray]:
    """Applique Tesseract sur une image ROI et retourne (texte, confiance, debug).

    L'implémentation suit la documentation de Tesseract : on fournit un texte
    noir sur fond blanc, on choisit explicitement ``--psm`` et ``--oem`` puis on
    limite le jeu de caractères attendu via ``tessedit_char_whitelist``.
    """

    params = params or DEFAULT_PARAMS
    gray, prepared = _prepare_image(bgr_or_gray, params)
    config = _build_config(params)
    lang = params.lang.strip() or DEFAULT_PARAMS.lang

    try:
        data = pytesseract.image_to_data(
            prepared,
            lang=lang,
            config=config,
            output_type=Output.DICT,
        )
    except TesseractNotFoundError:
        # L'appelant gère cette erreur pour alerter l'utilisateur.
        raise
    except TesseractError:
        empty = {"text": [], "conf": [], "level": []}
        dbg = _render_debug(prepared, empty, None, 0.0)
        return None, 0.0, dbg

    text = _extract_numeric_text(data, params.allow_decimal)
    confidence = _average_confidence(data)
    dbg = _render_debug(prepared, data, text, confidence)
    return text, confidence, dbg

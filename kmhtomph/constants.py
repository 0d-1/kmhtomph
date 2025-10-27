"""
Constantes et paramètres par défaut pour kmhtomph.
Séparées du reste pour éviter les imports circulaires et centraliser le tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

# Conversion
KMH_TO_MPH: float = 0.621371192237334  # 1 km/h = 0.62137 mph

# Texte incrusté (overlay) par défaut
DEFAULT_FONT_FAMILY: str = "DejaVu Sans"
DEFAULT_FONT_POINT_SIZE: int = 28
DEFAULT_TEXT_PADDING_PX: int = 6
DEFAULT_OUTLINE_THICKNESS_PX: int = 2
DEFAULT_FILL_OPACITY: float = 0.75  # 0..1 pour l'arrière-plan

# Anti-jitter / sélection de la meilleure valeur OCR
@dataclass(frozen=True)
class AntiJitterConfig:
    """
    Paramètres de lissage/anti-sauts pour la vitesse.
    - window_size: taille de la fenêtre du médian (en frames)
    - max_delta_kmh: saut max accepté d’une frame à l’autre (km/h), sinon on rejette
    - min_confidence: score mini pour accepter la valeur (0..1)
    - hold_max_gap_frames: nb de frames à “tenir” la dernière valeur quand tout rate
    """
    window_size: int = 5
    # Ajusté pour des FPS typiques (25–60) : 2.5 km/h par frame est bien plus réaliste que 8.0
    max_delta_kmh: float = 2.5
    min_confidence: float = 0.5
    # NEW: maintien configurable de la dernière valeur quand on a des trous
    hold_max_gap_frames: int = 6


DEFAULT_ANTI_JITTER = AntiJitterConfig()

# Débogage
DEFAULT_SHOW_DEBUG_THUMB: bool = True  # affiche la vignette OCR quand dispo
DEFAULT_DEBUG_THUMB_SIZE: int = 160    # pixels (carré)

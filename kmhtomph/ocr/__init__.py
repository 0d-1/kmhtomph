
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

from .tesseract import tesseract_ocr, TesseractParams, DEFAULT_PARAMS as DEFAULT_TESS_PARAMS
from .sevenseg import sevenseg_ocr
from .chooser import AntiJitterState, choose_best_kmh, reset as reset_state
from ..constants import DEFAULT_ANTI_JITTER, AntiJitterConfig


@dataclass
class OCRPipeline:
    anti_jitter: AntiJitterConfig = field(default_factory=lambda: DEFAULT_ANTI_JITTER)
    state: AntiJitterState = field(default_factory=lambda: AntiJitterState())
    tesseract_params: TesseractParams = field(default_factory=lambda: DEFAULT_TESS_PARAMS)

    def reset(self):
        reset_state(self.state)

    def read_kmh(self, roi_bgr: np.ndarray, mode: str = "auto"):
        """
        Returns (kmh_value:Optional[float], debug_bgr:Optional[np.ndarray], score:float, details:str)
        """
        candidates: List[Tuple[Optional[float], float, str, Optional[np.ndarray]]] = []
        dbg_best = None

        if mode in ("sevenseg", "auto"):
            txt7, conf7, dbg7 = sevenseg_ocr(roi_bgr)
            v7 = float(txt7) if txt7 is not None and txt7.strip() else None
            candidates.append((v7, float(conf7), "7seg", dbg7))

        if mode in ("tesseract", "auto"):
            txt, conf, dbg = tesseract_ocr(roi_bgr, self.tesseract_params)
            v = float(txt) if txt is not None and txt.strip() else None
            candidates.append((v, float(conf), "tess", dbg))

        simple = [(v, c, s) for (v, c, s, d) in candidates]
        chosen = choose_best_kmh(simple, self.state, self.anti_jitter)

        if chosen is None:
            # choose a dbg to show
            for v,c,s,d in candidates:
                if s == "tess" and d is not None:
                    dbg_best = d; break
                if d is not None:
                    dbg_best = d
            return None, dbg_best, 0.0, "no-accept"

        # find confidence from the chosen source
        conf_src = 0.0
        for v,c,s,d in candidates:
            if v is not None and abs(v - chosen) < 1e-6:
                conf_src = float(c)
                if d is not None: dbg_best = d
                break
        if dbg_best is None and candidates:
            dbg_best = candidates[0][3]

        return float(chosen), dbg_best, conf_src, "ok"

"""
Boîte de dialogue simple pour configurer le chemin de Tesseract.

Expose :
- class SettingsDialog(QDialog)
    .tesseract_path -> str
    .tesseract_params -> TesseractParams
Usage :
    dlg = SettingsDialog(self, initial_path=current_path, initial_params=params)
    if dlg.exec_() == QDialog.Accepted:
        save(dlg.tesseract_path, dlg.tesseract_params)
"""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSpinBox, QFormLayout
)

from ..ocr.tesseract import auto_locate_tesseract, TesseractParams, DEFAULT_PARAMS as DEFAULT_TESS_PARAMS


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        initial_path: Optional[str] = None,
        initial_params: Optional[TesseractParams] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Paramètres OCR")
        self.setModal(True)

        self._initial_params = initial_params or DEFAULT_TESS_PARAMS

        self._edit = QLineEdit(self)
        if initial_path:
            self._edit.setText(initial_path)

        pick_btn = QPushButton("Parcourir…", self)
        test_btn = QPushButton("Tester", self)
        ok_btn = QPushButton("OK", self)
        cancel_btn = QPushButton("Annuler", self)

        pick_btn.clicked.connect(self._on_pick)
        test_btn.clicked.connect(self._on_test)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(QLabel("Chemin Tesseract :", self))
        row.addWidget(self._edit)
        row.addWidget(pick_btn)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(test_btn)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)

        # Prétraitements
        pre_box = QGroupBox("Prétraitement de l'image", self)
        pre_lay = QVBoxLayout()

        self._chk_denoise = QCheckBox(
            "Filtre bilatéral : lisse le bruit mais ralentit sensiblement le traitement.",
            self,
        )
        self._chk_denoise.setChecked(self._initial_params.denoise_bilateral)

        self._chk_clahe = QCheckBox(
            "CLAHE (contraste adaptatif) : rehausse les zones sombres au prix de calculs supplémentaires.",
            self,
        )
        self._chk_clahe.setChecked(self._initial_params.clahe)

        self._chk_unsharp = QCheckBox(
            "Netteté renforcée : accentue les contours mais ajoute du coût de traitement.",
            self,
        )
        self._chk_unsharp.setChecked(self._initial_params.unsharp)

        scale_form = QFormLayout()
        self._spin_scale = QSpinBox(self)
        self._spin_scale.setRange(24, 400)
        self._spin_scale.setSuffix(" px")
        self._spin_scale.setValue(int(self._initial_params.scale_to_height))
        scale_form.addRow("Hauteur cible :", self._spin_scale)
        scale_desc = QLabel(
            "Une hauteur plus élevée améliore la lisibilité des petits chiffres mais augmente le temps de calcul.",
            self,
        )
        scale_desc.setWordWrap(True)

        pre_lay.addWidget(self._chk_denoise)
        pre_lay.addWidget(self._chk_clahe)
        pre_lay.addWidget(self._chk_unsharp)
        pre_lay.addLayout(scale_form)
        pre_lay.addWidget(scale_desc)
        pre_box.setLayout(pre_lay)

        morph_box = QGroupBox("Morphologie", self)
        morph_lay = QVBoxLayout()
        self._chk_morph_open = QCheckBox(
            "Ouverture : supprime les petits points blancs isolés (peut rogner les traits fins).",
            self,
        )
        self._chk_morph_open.setChecked(self._initial_params.morph_open)
        self._chk_morph_close = QCheckBox(
            "Fermeture : comble les petits trous noirs (lisse les bords mais peut épaissir le texte).",
            self,
        )
        self._chk_morph_close.setChecked(self._initial_params.morph_close)
        morph_lay.addWidget(self._chk_morph_open)
        morph_lay.addWidget(self._chk_morph_close)
        morph_box.setLayout(morph_lay)

        combos_box = QGroupBox("Combinaisons supplémentaires", self)
        combos_lay = QVBoxLayout()
        self._chk_psm8 = QCheckBox(
            "Essayer aussi le mode PSM 8 (bloc horizontal) après l'essai principal.",
            self,
        )
        self._chk_psm8.setChecked(self._initial_params.try_psm8)
        self._chk_psm6 = QCheckBox(
            "Essayer le mode PSM 6 (paragraphe) en dernier recours pour les mises en page inhabituelles.",
            self,
        )
        self._chk_psm6.setChecked(self._initial_params.try_psm6)
        self._chk_scale = QCheckBox(
            "Tester un redimensionnement à 140 px de hauteur si l'image initiale est plus petite.",
            self,
        )
        self._chk_scale.setChecked(self._initial_params.try_larger_scale)
        self._chk_nomorph = QCheckBox(
            "Essayer une variante sans opérations morphologiques (utile si le texte est déjà propre).",
            self,
        )
        self._chk_nomorph.setChecked(self._initial_params.try_without_morphology)
        combos_lay.addWidget(self._chk_psm8)
        combos_lay.addWidget(self._chk_psm6)
        combos_lay.addWidget(self._chk_scale)
        combos_lay.addWidget(self._chk_nomorph)
        combos_box.setLayout(combos_lay)

        bin_box = QGroupBox("Variantes de binarisation", self)
        bin_lay = QVBoxLayout()
        self._chk_adapt = QCheckBox(
            "Si l'image semble presque vide, tester une binarisation adaptative pour récupérer un contraste local.",
            self,
        )
        self._chk_adapt.setChecked(self._initial_params.try_adaptive_if_blank)
        self._chk_raw = QCheckBox(
            "Ajouter une version binaire brute (sans morphologie) pour comparer rapidement, au prix d'un appel Tesseract supplémentaire.",
            self,
        )
        self._chk_raw.setChecked(self._initial_params.try_raw_binarize)
        self._chk_comp = QCheckBox(
            "Quand seule l'ouverture ou la fermeture est active, tester aussi l'opération complémentaire (lent mais aide sur certains cas).",
            self,
        )
        self._chk_comp.setChecked(self._initial_params.try_complementary_morph)
        bin_lay.addWidget(self._chk_adapt)
        bin_lay.addWidget(self._chk_raw)
        bin_lay.addWidget(self._chk_comp)
        bin_box.setLayout(bin_lay)

        self._chk_raw.stateChanged.connect(self._sync_binary_options)
        self._sync_binary_options()

        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(pre_box)
        lay.addWidget(morph_box)
        lay.addWidget(combos_box)
        lay.addWidget(bin_box)
        lay.addLayout(btns)
        self.setLayout(lay)
        self.resize(680, 640)

    @property
    def tesseract_path(self) -> str:
        return self._edit.text().strip()

    @property
    def tesseract_params(self) -> TesseractParams:
        return replace(
            self._initial_params,
            denoise_bilateral=self._chk_denoise.isChecked(),
            clahe=self._chk_clahe.isChecked(),
            unsharp=self._chk_unsharp.isChecked(),
            scale_to_height=int(self._spin_scale.value()),
            morph_open=self._chk_morph_open.isChecked(),
            morph_close=self._chk_morph_close.isChecked(),
            try_psm8=self._chk_psm8.isChecked(),
            try_psm6=self._chk_psm6.isChecked(),
            try_larger_scale=self._chk_scale.isChecked(),
            try_without_morphology=self._chk_nomorph.isChecked(),
            try_adaptive_if_blank=self._chk_adapt.isChecked(),
            try_raw_binarize=self._chk_raw.isChecked(),
            try_complementary_morph=self._chk_comp.isChecked(),
        )

    def _sync_binary_options(self) -> None:
        self._chk_comp.setEnabled(self._chk_raw.isChecked())

    def _on_pick(self):
        start = self.tesseract_path or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Sélectionner l'exécutable Tesseract", start)
        if path:
            self._edit.setText(path)

    def _on_test(self):
        path = self.tesseract_path or None
        try:
            auto_locate_tesseract(path)
            # si OK, pytesseract utilisera ce chemin au prochain appel ; on vérifie l'existence
            if path and not os.path.exists(path):
                raise FileNotFoundError(path)
            QMessageBox.information(self, "Test Tesseract", "Configuration appliquée.\nUn test réel se fera à la première lecture OCR.")
        except Exception as e:
            QMessageBox.warning(self, "Test Tesseract", f"Échec de configuration : {e}")

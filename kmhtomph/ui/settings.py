"""Boîte de dialogue pour configurer Tesseract de manière simple et robuste."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..ocr.tesseract import (
    DEFAULT_PARAMS as DEFAULT_TESS_PARAMS,
    TesseractParams,
    auto_locate_tesseract,
)


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

        options_box = QGroupBox("Options principales", self)
        form = QFormLayout()

        self._lang_edit = QLineEdit(self)
        self._lang_edit.setPlaceholderText("eng, fra, eng+fra…")
        self._lang_edit.setText(self._initial_params.lang)
        form.addRow("Langue(s) :", self._lang_edit)

        self._psm_spin = QSpinBox(self)
        self._psm_spin.setRange(0, 13)
        self._psm_spin.setValue(int(self._initial_params.psm))
        self._psm_spin.setToolTip("Choisit le mode de segmentation de page (--psm).")
        form.addRow("Mode PSM :", self._psm_spin)

        self._oem_combo = QComboBox(self)
        oem_items = [
            ("Legacy (0)", 0),
            ("LSTM (1)", 1),
            ("Legacy+LSTM (2)", 2),
            ("Auto (3)", 3),
        ]
        for label, value in oem_items:
            self._oem_combo.addItem(label, value)
        current_oem = int(self._initial_params.oem)
        idx = next((i for i, (_, val) in enumerate(oem_items) if val == current_oem), 0)
        self._oem_combo.setCurrentIndex(idx)
        form.addRow("Moteur OEM :", self._oem_combo)

        self._allow_decimal = QCheckBox("Autoriser un séparateur décimal (.,)", self)
        self._allow_decimal.setChecked(self._initial_params.allow_decimal)
        form.addRow(self._allow_decimal)

        self._whitelist_edit = QLineEdit(self)
        self._whitelist_edit.setText(self._initial_params.whitelist)
        self._whitelist_edit.setToolTip(
            "Restreint les caractères acceptés (tessedit_char_whitelist)."
        )
        form.addRow("Jeu de caractères :", self._whitelist_edit)

        self._scale_spin = QSpinBox(self)
        self._scale_spin.setRange(24, 400)
        self._scale_spin.setSuffix(" px")
        self._scale_spin.setValue(int(self._initial_params.scale_to_height))
        form.addRow("Hauteur cible :", self._scale_spin)

        self._tessdata_edit = QLineEdit(self)
        if self._initial_params.tessdata_dir:
            self._tessdata_edit.setText(self._initial_params.tessdata_dir)
        tessdata_btn = QPushButton("Parcourir…", self)
        tessdata_btn.clicked.connect(self._on_pick_tessdata)

        tessdata_row = QHBoxLayout()
        tessdata_row.addWidget(self._tessdata_edit)
        tessdata_row.addWidget(tessdata_btn)
        form.addRow("Répertoire tessdata :", tessdata_row)

        options_box.setLayout(form)

        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(options_box)
        lay.addStretch(1)
        lay.addLayout(btns)
        self.setLayout(lay)
        self.resize(640, 420)

    @property
    def tesseract_path(self) -> str:
        return self._edit.text().strip()

    @property
    def tesseract_params(self) -> TesseractParams:
        lang = self._lang_edit.text().strip() or DEFAULT_TESS_PARAMS.lang
        whitelist = self._whitelist_edit.text().strip() or DEFAULT_TESS_PARAMS.whitelist
        tessdata_dir = self._tessdata_edit.text().strip() or None
        return replace(
            self._initial_params,
            lang=lang,
            psm=int(self._psm_spin.value()),
            oem=int(self._oem_combo.currentData()),
            allow_decimal=self._allow_decimal.isChecked(),
            whitelist=whitelist,
            scale_to_height=int(self._scale_spin.value()),
            tessdata_dir=tessdata_dir,
        )

    def _on_pick(self) -> None:
        start = self.tesseract_path or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner l'exécutable Tesseract",
            start,
        )
        if path:
            self._edit.setText(path)

    def _on_pick_tessdata(self) -> None:
        start = self._tessdata_edit.text().strip() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Sélectionner le répertoire tessdata",
            start,
        )
        if directory:
            self._tessdata_edit.setText(directory)

    def _on_test(self) -> None:
        path = self.tesseract_path or None
        tessdata_dir = self._tessdata_edit.text().strip() or None

        try:
            auto_locate_tesseract(path)
            if path and not os.path.exists(path):
                raise FileNotFoundError(path)
            if tessdata_dir and not os.path.isdir(tessdata_dir):
                raise FileNotFoundError(tessdata_dir)
            QMessageBox.information(
                self,
                "Test Tesseract",
                "Configuration appliquée.\nUn test réel se fera lors de la prochaine lecture OCR.",
            )
        except Exception as e:  # pragma: no cover - interaction utilisateur
            QMessageBox.warning(self, "Test Tesseract", f"Échec de configuration : {e}")

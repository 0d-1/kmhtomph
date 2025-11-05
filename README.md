# kmh→mph OCR

kmh→mph OCR est une application de bureau Qt qui lit une vidéo de compteur de vitesse et produit une transcription propre de la vitesse en miles/heure. Le programme combine un pipeline d'OCR spécialisé (afficheur sept segments et Tesseract), un lissage anti-jitter et un export vidéo avec incrustation du texte mis en forme.

## Fonctionnalités

- **Lecture vidéo et navigation** : ouvre des fichiers vidéo via OpenCV, affiche la frame courante dans un canvas Qt et permet la lecture/pause et le déplacement dans la timeline avec une barre de progression détaillée.
- **Sélection précise de la zone OCR** : dessinez un quadrilatère correspondant à la zone contenant les chiffres. Le programme redresse la perspective par homographie avant l’analyse, ce qui améliore nettement la reconnaissance.
- **Pipeline OCR hybride** : combine un détecteur rapide pour afficheurs sept segments et Tesseract. Le mode « auto » choisit la meilleure valeur en fonction de la confiance et des heuristiques, avec un système anti-sauts configurable.
- **Détection automatique de Tesseract** : tente de localiser le binaire `tesseract` sur votre système (ou utilise un chemin personnalisé) et configure automatiquement `pytesseract` si trouvé.
- **Lissage et validation des vitesses** : filtre les valeurs aberrantes, applique une médiane centrée sur plusieurs frames et limite les variations abruptes pour produire une série temporelle cohérente.
- **Export vidéo avec overlay** : génère une vidéo MP4/AVI incluant le texte formaté (police, outline, fond semi-transparent) et expose des hooks pour personnaliser l'overlay ou suivre la progression.
- **Interface personnalisable** : enregistrez vos préférences (chemin Tesseract, style d’overlay, paramètres de lissage), activez une vignette debug de l’OCR et ajustez l’affichage du texte.

## Installation

1. Installez les dépendances système nécessaires (OpenCV, Qt5, Tesseract). Sous Debian/Ubuntu :
   ```bash
   sudo apt install python3-pyqt5 python3-opencv tesseract-ocr
   ```
2. Créez et activez un environnement virtuel Python ≥ 3.9 puis installez les bibliothèques Python :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install PyQt5 opencv-python numpy pytesseract
   ```
3. Si Tesseract n’est pas sur le `PATH`, ouvrez l’application et renseignez son chemin exact via **Paramètres → Tesseract…**.

## Utilisation

```bash
python -m kmhtomph.app /chemin/vers/video.mp4
```

1. **Choisir la zone** : cliquez sur « Tracer zone de sortie » et ajustez les coins du quadrilatère autour de l’affichage km/h. La zone est redressée automatiquement pour l’OCR.
2. **Sélectionner le mode OCR** : laissez « auto » pour le mode hybride ou forcez « sevenseg »/« tesseract » selon la nature de l’affichage.
3. **Lancer l’analyse** : démarrez la lecture. La vitesse instantanée et la conversion en mph apparaissent en haut à droite, avec une vignette debug facultative.
4. **Exporter** : utilisez « Exporter… » pour générer une vidéo avec texte incrusté ou une feuille de temps (CSV) selon les options choisies.

Les valeurs acceptées sont mémorisées dans une timeline consultable (tableau sous la vidéo), et vous pouvez corriger manuellement un segment si nécessaire avant export.

## Structure du projet

```
kmhtomph/
├── app.py              # Point d’entrée Qt
├── constants.py        # Constantes et configuration globale
├── ocr/                # Pipeline OCR (Tesseract, sept segments, anti-jitter)
├── ui/                 # Widgets Qt : fenêtre principale, canvas, dialogues
└── video/              # Lecture, export et dessin des overlays
```

## Développement

- Exécutez `python -m kmhtomph.app` pour lancer l’application en mode développement avec la console interactive.
- Activez `DEFAULT_SHOW_DEBUG_THUMB` dans `constants.py` pour visualiser la vignette OCR pendant l’analyse.
- Les modules `video.overlay` et `video.exporter` sont découplés de Qt Widgets ; vous pouvez les réutiliser dans des scripts d’export automatisés ou des tests unitaires.

## Licence

Ce dépôt ne contient pas d’information de licence.

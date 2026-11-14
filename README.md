# Assistance à la lecture de texte dans l'espace public (personnes malvoyantes)

Pipeline Python / OpenCV / Tesseract : une photo de plaque, de numéro ou de panneau
en entrée, le texte lu en sortie.

## Organisation

```
data/images/         images de test (plaques/, numeros/, panneaux/, affiches/)
data/ground_truth/   vérité terrain : un <nom_image>.txt par image
data/SOURCES.csv     auteur et licence de chaque image (créé par telecharger_commons.py)
src/                 modules du pipeline (importés par les scripts)
scripts/             scripts exécutables
results/             figures et CSV produits
```

## Installation

### Option A : en local avec uv (Python 3.14)

```bash
uv sync
```

Tesseract n'est pas un paquet Python : installez-le à part, avec les langues fra (et ara si besoin).

- Ubuntu/Debian : `sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
- macOS : `brew install tesseract tesseract-lang`
- Windows : installeur UB Mannheim, puis ajouter `tesseract.exe` au PATH

### Option B : avec Docker (Tesseract déjà inclus)

```bash
docker compose build
```

## Utilisation

| Action                      | uv (local)                                       | Docker                                                |
|-----------------------------|--------------------------------------------------|------------------------------------------------------|
| Télécharger les images      | `uv run scripts/telecharger_commons.py --preset` | `docker compose run --rm ocr scripts/telecharger_commons.py --preset` |
| Comparer les prétraitements | `uv run scripts/etape2_pretraitement.py --ocr`   | `docker compose run --rm ocr scripts/etape2_pretraitement.py --ocr`   |

## Avancement

| Étape                                    | Module                  | Statut  |
|------------------------------------------|-------------------------|---------|
| 1. Acquisition                           | `src/acquisition.py`    | fait    |
| 2. Prétraitement (débruitage, contraste) | `src/preprocessing.py`  | fait    |
| 3. Binarisation                          | `src/binarization.py`   | à venir |
| 4. Localisation du texte                 | `src/text_detection.py` | à venir |
| 5. OCR                                   | `src/ocr.py`            | à venir |
| 6. Restitution                           | `src/restitution.py`    | à venir |
| 7. Évaluation / benchmark                | `src/evaluation.py`     | à venir |

## Sorties de l'étape 2 (`results/etape2/`)

- `figures/<image>_debruitage.png` : les 5 débruitages côte à côte
- `figures/<image>_zoom.png` : zoom sur la zone de texte
- `figures/<image>_contraste.png` : les 5 méthodes de contraste avec leurs histogrammes
- `debruitage_reference.csv` : PSNR, SSIM et temps sous bruit ajouté. Seules les images
  déjà propres (σ estimé ≤ 6) servent de référence.
- `contraste.csv` : contraste RMS, entropie et netteté
- `ocr_pretraitement.csv` : CER de Tesseract pour chaque combinaison (option `--ocr`)
- `synthese_*.png` : moyennes sur toute la base

Le CER peut dépasser 1 quand Tesseract lit des caractères parasites dans le fond :
la médiane est donc affichée en plus de la moyenne.

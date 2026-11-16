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

OpenCV est volontairement limité à la version 4 (`opencv-python-headless<5` dans
`pyproject.toml`) : dans OpenCV 5.0.0, le détecteur MSER ne trouve aucune région,
même sur un texte parfaitement net.

Tesseract n'est pas un paquet Python : installez-le à part, avec les langues fra (et ara si besoin).

- Ubuntu/Debian : `sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
- macOS : `brew install tesseract tesseract-lang`
- Windows : installeur UB Mannheim, puis ajouter `tesseract.exe` au PATH

### Option B : avec Docker (Tesseract déjà inclus)

```bash
docker compose build
```

## Utilisation

| Action | uv (local) | Docker |
|---|---|---|
| Télécharger les images | `uv run scripts/telecharger_commons.py --preset` | `docker compose run --rm ocr scripts/telecharger_commons.py --preset` |
| Comparer les prétraitements | `uv run scripts/etape2_pretraitement.py --ocr` | `docker compose run --rm ocr scripts/etape2_pretraitement.py --ocr` |
| Mesurer la baseline (Tesseract sur l'image brute) | `uv run scripts/baseline_tesseract.py` | `docker compose run --rm ocr scripts/baseline_tesseract.py` |
| Comparer les binarisations | `uv run scripts/etape3_binarisation.py --ocr` | `docker compose run --rm ocr scripts/etape3_binarisation.py --ocr` |
| Localiser le texte et lire chaque région | `uv run scripts/etape4_localisation.py` | `docker compose run --rm ocr scripts/etape4_localisation.py` |
| Créer les fichiers de vérité terrain | `bash scripts/creer_verite_terrain.sh` | |

## Avancement

| Étape | Module | Statut |
|---|---|---|
| 1. Acquisition | `src/acquisition.py` | fait |
| 2. Prétraitement (débruitage, contraste) | `src/preprocessing.py` | fait |
| 3. Binarisation | `src/binarization.py` | fait |
| 4. Localisation du texte | `src/text_detection.py` | fait |
| 5. OCR par région | `src/ocr.py` | fait |
| 6. Restitution | `src/restitution.py` | à venir |
| 7. Évaluation / benchmark | `src/evaluation.py` | métriques faites, benchmark final à venir |

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

## Sorties de l'étape 3 (`results/etape3/`)

- `figures/<image>_binarisation.png` : les 7 binarisations côte à côte
- `figures/<image>_zoom.png` : zoom sur la zone de texte
- `figures/<image>_histogramme.png` : histogramme avec les seuils fixe, Otsu et triangle
- `binarisation.csv` : temps, proportion de pixels texte, nombre de composantes, CER, WER
  et texte lu, pour chaque méthode. La ligne « gris (Tesseract) » est la référence :
  Tesseract binarise lui-même l'image.
- `balayage_sauvola.csv` (option `--balayage`) : CER de Sauvola pour chaque fenêtre et chaque k

Options utiles : `--fenetre 31` (taille de la fenêtre locale), `--polarite clair|fonce`
(forcer la polarité), `--sans-nettoyage`, `--debruitage` / `--contraste`
(prétraitement appliqué avant la binarisation).

## Sorties de l'étape 4 (`results/etape4/`)

- `figures/<image>_detection.png` : boîtes trouvées par la morphologie et par MSER
  (rouge fin : candidats rejetés par la vérification ; vert : régions retenues)
- `figures/<image>_regions.png` : régions binarisées envoyées à Tesseract, avec le texte lu
  et la confiance
- `textes/<image>.txt` : texte restitué
- `localisation.csv` : scores par image et par configuration
- `synthese_localisation.csv` et `synthese_*.png` : médianes par configuration, avec la
  ligne « sans_localisation » (= baseline psm 11) pour comparaison

Options utiles : `--detecteurs morpho mser union`, `--binarisations aucune otsu sauvola`,
`--confiance 50` (confiance minimale par mot), `--config-planches morpho+otsu`,
`--sans-reference` (ne pas recalculer la baseline, plus rapide).

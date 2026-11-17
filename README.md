# Assistance à la lecture de texte dans l'espace public (personnes malvoyantes)

Une photo de plaque de rue, de panneau, de numéro ou d'enseigne en entrée ; le texte lu,
restitué de façon accessible, en sortie. Python, OpenCV, NumPy, SciPy, Matplotlib et Tesseract.

```bash
uv run scripts/lire_image.py photo.jpg --carte carte.png
```

## Résultats (50 images annotées)

| | Baseline (Tesseract sur l'image brute) | Configuration finale (MSER + Otsu, lecture zone par zone) |
|---|---|---|
| F1 médian des mots | 0,013 | 0,281 (IC 95 % : 0,12 à 0,41) |
| Précision médiane | 0,007 | 0,250 |
| Rappel médian | 0,120 | 0,400 |
| Images améliorées / égales / dégradées | | 32 / 14 / 4 (Wilcoxon p < 0,001) |
| Temps moyen par image | 4,3 s | 5,5 s (19 s au pire) |

L'étape décisive est la **localisation du texte** : sur l'image entière, Tesseract invente
des lettres dans les arbres et les façades (5 586 mots lus pour 326 attendus), et ni le
débruitage ni la binarisation de toute l'image n'y changent grand-chose.
Détails, justifications et analyse des échecs : `docs/rapport/rapport.pdf`.

## Organisation

```
src/                    modules du pipeline
  acquisition.py          1. chargement, niveaux de gris, côté max 1600 px
  preprocessing.py        2. débruitage (gaussien, médian, bilatéral, NLM), contraste (CLAHE…)
  binarization.py         3. Otsu, triangle, adaptatives, Niblack, Sauvola, polarité, nettoyage
  text_detection.py       4. localisation : MSER, morphologie, vérification, pyramide
  ocr.py                  5. OCR par région (Tesseract psm 7, filtre de confiance)
  restitution.py          6. post-traitement, affichage, carte de lecture fort contraste
  evaluation.py           7. CER, WER, précision / rappel / F1 des mots
  metrics_image.py        PSNR, SSIM, contraste, netteté
  visualization.py        figures Matplotlib
scripts/
  telecharger_commons.py  constitution de la base (licences libres, SOURCES.csv)
  creer_verite_terrain.sh crée les fichiers de vérité terrain vides
  lire_image.py           L'APPLICATION : image(s) → texte + carte de lecture
  baseline_tesseract.py   baseline : Tesseract sur l'image brute
  etape2_pretraitement.py comparaison débruitage × contraste
  etape3_binarisation.py  comparaison des binarisations
  etape4_localisation.py  comparaison des détecteurs + OCR par région
  benchmark.py            synthèse finale : tableaux LaTeX, figures, statistiques
data/
  images/                 50 images de test (affiches, numeros, panneaux, plaques)
  ground_truth/           vérité terrain : un <image>.txt par image
  SOURCES.csv             auteur et licence de chaque image
results/                  sorties des scripts (results/benchmark/ alimente le rapport)
docs/
  rapport/                rapport technique (rapport.tex, rapport.pdf)
  presentation/           présentation Beamer (presentation.tex, presentation.pdf)
  figures/                figures fixes du rapport et de la présentation
Dockerfile, compose.yml   exécution sans rien installer d'autre que Docker
```

## Installation

Deux options, au choix.

### Option A : en local avec uv (Python 3.14)

```bash
uv sync
```

OpenCV est volontairement limité à la version 4 (`opencv-python-headless<5`) : dans
OpenCV 5.0.0, le détecteur MSER ne trouve aucune région, même sur un texte parfaitement net.

Tesseract s'installe à part, avec les langues fra (et ara si besoin) :

- Ubuntu/Debian/WSL : `sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
- macOS : `brew install tesseract tesseract-lang`
- Windows : installeur UB Mannheim, puis ajouter `tesseract.exe` au PATH

### Option B : Docker (Tesseract inclus)

```bash
docker compose build
docker compose run --rm ocr -c "import cv2, pytesseract; print(cv2.__version__, pytesseract.get_tesseract_version())"
```

La seconde commande vérifie l'image : elle doit afficher OpenCV 4.14 et Tesseract 5.x.

L'image contient le code ; `data/` et `results/` restent sur votre disque et sont montés
dans le conteneur. Deux règles en découlent :

- **les sorties doivent aller dans `results/`** : un fichier écrit ailleurs dans le
  conteneur disparaît à la fin de la commande ;
- **une photo hors de `data/`** doit être montée explicitement (voir plus bas).

Le conteneur tourne avec votre identifiant d'utilisateur (1000 par défaut, celui du premier
utilisateur sous WSL et Linux) : les fichiers créés dans `results/` vous appartiennent. Si
votre identifiant est différent, lancez d'abord `export UID GID=$(id -g)`.

Après une modification de `src/`, `scripts/`, `pyproject.toml` ou `uv.lock`, reconstruisez
l'image avec `docker compose build`.

## Utilisation

### Lire une image ou un dossier

```bash
# en local
uv run scripts/lire_image.py photo.jpg                       # affichage console
uv run scripts/lire_image.py photo.jpg --carte carte.png     # + carte de lecture PNG
uv run scripts/lire_image.py data/images/ --sortie results/lectures   # un .txt + une carte par image

# avec Docker
docker compose run --rm ocr scripts/lire_image.py data/images/panneaux/panneaux_006.jpg \
    --carte results/carte_006.png
docker compose run --rm ocr scripts/lire_image.py data/images/ --sortie results/lectures

# avec Docker, une photo située ailleurs (ici dans ~/photos)
docker compose run --rm -v ~/photos:/photos ocr scripts/lire_image.py /photos/rue.jpg \
    --carte results/rue_carte.png
```

Options : `--detecteur mser|morpho|union`, `--binarisation aucune|otsu|sauvola`,
`--canaux b_lab clahe` (texte coloré ou peu contrasté ; environ deux fois plus lent),
`--langue fra+eng+ara`, `--confiance 50`, `--max-regions 40`.

La carte de lecture affiche le texte en jaune sur noir (contraste d'environ 17:1, niveau AAA
des WCAG), en très grand, une information par ligne ; chaque ligne porte le numéro de sa zone
sur la photo.

### Reproduire l'évaluation

| Étape | uv (local) | Docker |
|---|---|---|
| Baseline | `uv run scripts/baseline_tesseract.py` | `docker compose run --rm ocr scripts/baseline_tesseract.py` |
| Prétraitement | `uv run scripts/etape2_pretraitement.py --ocr` | `docker compose run --rm ocr scripts/etape2_pretraitement.py --ocr` |
| Binarisation | `uv run scripts/etape3_binarisation.py --ocr` | `docker compose run --rm ocr scripts/etape3_binarisation.py --ocr` |
| Localisation (v1) | `uv run scripts/etape4_localisation.py --hauteur-min 0.008 --max-regions 0` | `docker compose run --rm ocr scripts/etape4_localisation.py --hauteur-min 0.008 --max-regions 0` |
| Localisation (v2, finale) | `uv run scripts/etape4_localisation.py --detecteurs mser --binarisations otsu aucune --sans-reference --sortie results/etape4_v2` | même script, précédé de `docker compose run --rm ocr` |
| Benchmark | `uv run scripts/benchmark.py --finale etape4_v2:mser+otsu` | `docker compose run --rm ocr scripts/benchmark.py --finale etape4_v2:mser+otsu` |

Chaque dossier `results/etape4*` est traité comme une variante par le benchmark ;
`--finale` désigne la configuration retenue (`<dossier>:<configuration>`). La version 1
correspond aux réglages d'origine (hauteur minimale de ligne 0,8 %, pas de plafond de zones),
la version 2 aux réglages actuels par défaut (0,5 %, 40 zones au plus).

L'image Docker utilise le Tesseract de Debian Bookworm (5.3). Si votre Tesseract local est
d'une autre version, les scores peuvent varier légèrement entre les deux environnements :
comparez toujours des résultats obtenus dans le même.

### Métriques

- **CER / WER** : distance de Levenshtein en caractères / en mots, rapportée à la référence.
  Non bornés par 1 : chaque caractère inventé compte.
- **Précision, rappel, F1 des mots** (métriques principales), sans tenir compte de l'ordre.
- Médianes, IC à 95 % par bootstrap, test apparié de Wilcoxon contre la baseline.

## Rapport et présentation

Le rapport et la présentation incluent directement les tableaux, figures et chiffres de
`results/benchmark/`. Après un nouveau benchmark, il suffit de recompiler.

En local (paquets : `latexmk texlive-latex-extra texlive-lang-french texlive-fonts-recommended`) :

```bash
cd docs/rapport && latexmk -pdf rapport.tex && cd ../..
cd docs/presentation && latexmk -pdf presentation.tex && cd ../..
```

Avec Docker, sans installer LaTeX (image TeX Live complète de plusieurs Go, téléchargée au
premier lancement) :

```bash
docker compose run --rm docs
```

## Limites connues

Écriture arabe (le détecteur suppose des lettres séparées ; Tesseract lit « قف » quand on lui
donne la zone et `--langue fra+eng+ara`), lettres gravées, néons de nuit, texte de moins
d'environ 8 px de haut. Huit vérités terrain contiennent de l'arabe : avec `fra+eng`, ces
mots sont toujours comptés comme manqués. Voir la section « Analyse des échecs » du rapport.

## Données et licences

Images issues de Wikimedia Commons sous licences CC0, CC BY ou CC BY-SA ; l'auteur, la
licence et la page source de chaque image figurent dans `data/SOURCES.csv`. Les images
modifiées par le pipeline (binarisations, figures) restent soumises à ces licences.

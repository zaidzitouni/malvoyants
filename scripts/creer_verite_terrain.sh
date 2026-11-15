#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# creer_verite_terrain.sh
# Crée un fichier texte vide data/ground_truth/<nom_image>.txt pour chaque image
# de data/images/ (sous-dossiers compris). Il reste ensuite à y taper le texte
# visible sur l'image.
#
# Les fichiers déjà existants ne sont PAS modifiés : on peut relancer le script
# sans risque après avoir ajouté des images.
#
# Usage (depuis n'importe quel dossier) :
#   bash scripts/creer_verite_terrain.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# Se placer à la racine du projet (le dossier parent de scripts/),
# quel que soit le dossier depuis lequel on lance le script.
cd "$(dirname "$0")/.."

IMAGES="data/images"
VERITE="data/ground_truth"

if [ ! -d "$IMAGES" ]; then
  echo "Dossier $IMAGES introuvable : lancez d'abord telecharger_commons.py" >&2
  exit 1
fi
mkdir -p "$VERITE"

crees=0
existants=0
# find gère les images à n'importe quelle profondeur et les extensions en
# majuscules (.JPG, .PNG) ou en .jpeg, que la boucle *.{jpg,png} manquerait.
while IFS= read -r -d '' image; do
  nom="$(basename "${image%.*}")"
  cible="$VERITE/$nom.txt"
  if [ -e "$cible" ]; then
    existants=$((existants + 1))
  else
    : > "$cible"
    crees=$((crees + 1))
  fi
done < <(find "$IMAGES" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

nb_images=$(find "$IMAGES" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
vides=$(find "$VERITE" -name '*.txt' -empty | wc -l)

echo "Images trouvées      : $nb_images"
echo "Fichiers créés       : $crees"
echo "Déjà existants       : $existants"
echo "Encore à remplir     : $vides (fichiers vides dans $VERITE)"
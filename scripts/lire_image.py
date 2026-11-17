#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lire_image.py — L'application : une photo en entrée, le texte lu en sortie.

Enchaîne tout le pipeline avec la configuration retenue à l'issue des
comparaisons (étapes 2 à 4) :
  1. acquisition       : chargement, niveaux de gris, côté max 1600 px
  2. (prétraitement)   : non appliqué à l'image entière (aucun gain mesuré) ;
                         la binarisation se fait région par région
  3-4. localisation    : MSER multi-échelles + vérification par segmentation
                         des caractères
  5. OCR par région    : recadrage, mise à l'échelle, binarisation d'Otsu,
                         Tesseract psm 7, filtre de confiance
  6. restitution       : texte nettoyé, affichage console, carte de lecture

Usage (depuis la racine du projet) :
  uv run scripts/lire_image.py data/images/panneaux/panneaux_006.jpg
  uv run scripts/lire_image.py photo.jpg --carte resultat.png
  uv run scripts/lire_image.py photo.jpg --canaux b_lab clahe --langue fra+eng+ara
  uv run scripts/lire_image.py dossier/ --sortie results/lectures
"""

import argparse
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from acquisition import EXTENSIONS, acquerir, lister_images  # noqa: E402
from ocr import lire_regions  # noqa: E402
from restitution import afficher_console, carte_de_lecture, post_traiter  # noqa: E402
from text_detection import (CANAUX_DISPONIBLES, HAUTEUR_MIN, MAX_REGIONS,  # noqa: E402
                            canaux_couleur, detecter_texte)


def lire(chemin, detecteur="mser", binarisation="otsu", canaux_noms=(), langue="fra+eng",
         confiance=50, hauteur_min=HAUTEUR_MIN, max_regions=MAX_REGIONS):
    """Pipeline complet. Renvoie (lectures [(ligne, boîte)], image couleur, durée en s)."""
    t0 = time.perf_counter()
    couleur, gris = acquerir(chemin)
    canaux = canaux_couleur(couleur, canaux_noms) if canaux_noms else None
    boites, _ = detecter_texte(gris, detecteur, canaux=canaux, hauteur_min=hauteur_min,
                               max_regions=max_regions)
    texte, details = lire_regions(gris, boites, binarisation, langue,
                                  confiance_min=confiance, canaux=canaux)
    lectures = post_traiter([(d["boite"], d["texte"]) for d in details])
    return lectures, couleur, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entree", help="une image ou un dossier d'images")
    p.add_argument("--carte", help="chemin du PNG « carte de lecture » (une seule image)")
    p.add_argument("--sortie", help="dossier : un .txt et une carte .png par image")
    p.add_argument("--detecteur", default="mser", choices=["mser", "morpho", "union"])
    p.add_argument("--binarisation", default="otsu", choices=["aucune", "otsu", "sauvola"])
    p.add_argument("--canaux", nargs="*", default=[], choices=CANAUX_DISPONIBLES)
    p.add_argument("--langue", default="fra+eng")
    p.add_argument("--confiance", type=float, default=50)
    p.add_argument("--max-regions", type=int, default=MAX_REGIONS)
    args = p.parse_args()

    entree = Path(args.entree)
    if entree.is_dir():
        images = lister_images(entree)
    elif entree.is_file() and entree.suffix.lower() in EXTENSIONS:
        images = [entree]
    else:
        sys.exit(f"Entrée introuvable ou non reconnue : {entree}")
    sortie = Path(args.sortie) if args.sortie else None
    if sortie:
        sortie.mkdir(parents=True, exist_ok=True)

    for chemin in images:
        lectures, couleur, duree = lire(chemin, args.detecteur, args.binarisation, args.canaux,
                                               args.langue, args.confiance,
                                               max_regions=args.max_regions or None)
        lignes = [ligne for ligne, _ in lectures]
        afficher_console(lignes, titre=f"{chemin.name}  ({duree:.1f} s)")
        if sortie:
            (sortie / f"{chemin.stem}.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")
            carte_de_lecture(couleur, lectures, sortie / f"{chemin.stem}_carte.png")
        elif args.carte:
            carte_de_lecture(couleur, lectures, args.carte)
            print(f"Carte de lecture : {args.carte}")
    if sortie:
        print(f"Textes et cartes de lecture : {sortie}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_tesseract.py — Point de départ : Tesseract sur l'image BRUTE.

Aucun traitement : ni redimensionnement, ni niveaux de gris, ni débruitage,
ni binarisation. L'image est passée telle quelle (couleur, résolution
d'origine) à Tesseract, qui applique sa propre binarisation interne (Otsu).
Tous les gains des étapes suivantes seront mesurés par rapport à ces scores.

Deux modes de segmentation Tesseract sont mesurés :
  psm 3   mode par défaut : suppose une page structurée (colonnes, paragraphes).
  psm 11  « texte épars » : cherche autant de texte que possible sans
          supposer de mise en page. A priori plus adapté aux scènes de rue.

Seules les images dont le fichier de vérité terrain est NON VIDE sont
évaluées. La catégorie est le préfixe du nom (panneaux_004 → panneaux).

Sorties dans results/baseline/ :
  baseline.csv           une ligne par image et par psm (scores + texte lu)
  synthese_baseline.csv  moyennes et médianes par catégorie et par psm
  baseline_<metrique>.png

Usage (depuis la racine du projet) :
  uv run scripts/baseline_tesseract.py
  uv run scripts/baseline_tesseract.py --psm 11 --max-images 10
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from acquisition import charger_image, lister_images  # noqa: E402
from evaluation import evaluer  # noqa: E402
from visualization import barres_metriques  # noqa: E402

METRIQUES = ("cer", "wer", "precision_mots", "rappel_mots", "f1_mots")


def categorie(chemin):
    return chemin.stem.split("_")[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(RACINE / "data" / "images"))
    p.add_argument("--verite", default=str(RACINE / "data" / "ground_truth"))
    p.add_argument("--sortie", default=str(RACINE / "results" / "baseline"))
    p.add_argument("--psm", nargs="+", default=["3", "11"], help="modes Tesseract à mesurer")
    p.add_argument("--langue", default="fra+eng")
    p.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    args = p.parse_args()

    import pytesseract

    # Images avec une vérité terrain remplie
    paires = []
    for chemin in lister_images(args.images):
        gt = Path(args.verite) / f"{chemin.stem}.txt"
        if gt.is_file() and gt.read_text(encoding="utf-8").strip():
            paires.append((chemin, gt.read_text(encoding="utf-8")))
    if args.max_images:
        paires = paires[:args.max_images]
    if not paires:
        sys.exit("Aucune image avec vérité terrain remplie.")
    print(f"{len(paires)} image(s) évaluée(s) | langue {args.langue} | psm {', '.join(args.psm)}")

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    resultats = defaultdict(lambda: defaultdict(list))  # (psm, catégorie) -> métrique -> valeurs

    with open(sortie / "baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "categorie", "largeur", "hauteur", "psm", *METRIQUES,
                    "temps_s", "nb_mots_attendus", "nb_mots_lus", "texte_ocr"])
        for n, (chemin, reference) in enumerate(paires, 1):
            # BGR (OpenCV) -> RGB : Tesseract attend du RGB. C'est la seule
            # conversion, purement technique, appliquée à l'image brute.
            image = cv2.cvtColor(charger_image(chemin), cv2.COLOR_BGR2RGB)
            h, l = image.shape[:2]
            ligne = [f"[{n}/{len(paires)}] {chemin.name:24s}"]
            for psm in args.psm:
                t0 = time.perf_counter()
                texte = pytesseract.image_to_string(image, lang=args.langue, config=f"--psm {psm}")
                dt = time.perf_counter() - t0
                s = evaluer(reference, texte)
                for m in METRIQUES:
                    resultats[(psm, categorie(chemin))][m].append(s[m])
                    resultats[(psm, "TOUTES")][m].append(s[m])
                w.writerow([chemin.name, categorie(chemin), l, h, psm,
                            *(f"{s[m]:.3f}" for m in METRIQUES), f"{dt:.2f}",
                            len(reference.split()), len(texte.split()), " ".join(texte.split())])
                ligne.append(f"psm{psm}: CER={s['cer']:.2f} rappel={s['rappel_mots']:.2f}")
            print("  ".join(ligne))

    # --- Synthèse ---------------------------------------------------------
    cles = sorted(resultats, key=lambda k: (k[0], k[1] == "TOUTES", k[1]))
    with open(sortie / "synthese_baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["psm", "categorie", "n"] + [f"{m}_{s}" for m in METRIQUES for s in ("moy", "med")])
        for psm, cat in cles:
            v = resultats[(psm, cat)]
            w.writerow([psm, cat, len(v["cer"])] +
                       [f"{fn(v[m]):.3f}" for m in METRIQUES for fn in (np.mean, np.median)])

    print("\n=== BASELINE : Tesseract sur l'image brute ===")
    print(f"  {'psm':4s} {'catégorie':12s} {'n':>3s} {'CER moy':>8s} {'CER méd':>8s} "
          f"{'WER méd':>8s} {'rappel méd':>11s} {'F1 méd':>7s}")
    for psm, cat in cles:
        v = resultats[(psm, cat)]
        print(f"  {psm:4s} {cat:12s} {len(v['cer']):3d} {np.mean(v['cer']):8.3f} "
              f"{np.median(v['cer']):8.3f} {np.median(v['wer']):8.3f} "
              f"{np.median(v['rappel_mots']):11.3f} {np.median(v['f1_mots']):7.3f}")

    for m, titre in (("cer", "CER médian (plus bas = mieux)"),
                     ("rappel_mots", "Rappel médian des mots (plus haut = mieux)")):
        barres_metriques({f"psm {psm} · {cat}": float(np.median(resultats[(psm, cat)][m]))
                          for psm, cat in cles},
                         f"Baseline — {titre}", m, sortie / f"baseline_{m}.png")

    print(f"\nRésultats : {sortie}")


if __name__ == "__main__":
    main()

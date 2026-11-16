#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etape3_binarisation.py — Comparaison des binarisations globales et adaptatives.

Pour chaque image de data/images/ :
  1. prétraitement (par défaut NLM + CLAHE, modifiable) ;
  2. détection de la polarité (texte clair sur fond foncé → inversion) ;
  3. les 7 binarisations : fixe, Otsu, triangle, adaptative moyenne,
     adaptative gaussienne, Niblack, Sauvola ;
  4. figures : grille des résultats, zoom sur la zone de texte, histogramme
     avec les seuils globaux ;
  5. (option --ocr) CER/WER de Tesseract sur chaque binarisation, comparés à
     la référence « gris » : Tesseract sur l'image en niveaux de gris, qu'il
     binarise lui-même en interne (Otsu). Cette ligne répond à la question :
     notre binarisation fait-elle mieux que celle de Tesseract ?

Option --balayage : CER de Sauvola pour plusieurs tailles de fenêtre et
valeurs de k, pour justifier les paramètres retenus dans le rapport.

Sorties dans results/etape3/ :
  figures/<image>_binarisation.png, _zoom.png, _histogramme.png
  binarisation.csv, synthese_ocr.csv, [balayage_sauvola.csv], synthese_*.png

Usage (depuis la racine du projet) :
  uv run scripts/etape3_binarisation.py --max-images 5
  uv run scripts/etape3_binarisation.py --ocr
  uv run scripts/etape3_binarisation.py --ocr --balayage --contraste egalisation
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

from acquisition import acquerir, lister_images  # noqa: E402
from binarization import (BINARISATION, GLOBALES,  # noqa: E402
                          normaliser_polarite, sauvola, nettoyer)
from evaluation import evaluer  # noqa: E402
from preprocessing import pretraiter  # noqa: E402
from visualization import (barres_metriques, grille_methodes,  # noqa: E402
                           histogramme_seuils, zoom_comparatif)

METRIQUES = ("cer", "wer", "precision_mots", "rappel_mots", "f1_mots")
FENETRES = (15, 31, 51, 81, 121)
VALEURS_K = (0.1, 0.2, 0.35, 0.5)


def zone_la_plus_texturee(gris, taille=200):
    taille = min(taille, *gris.shape)
    energie = np.abs(cv2.Laplacian(gris, cv2.CV_32F))
    somme = cv2.boxFilter(energie, -1, (taille, taille), normalize=False)
    m = taille // 2
    interieur = somme[m:gris.shape[0] - m or None, m:gris.shape[1] - m or None]
    y, x = np.unravel_index(np.argmax(interieur), interieur.shape)
    return int(x), int(y), taille, taille


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(RACINE / "data" / "images"))
    p.add_argument("--verite", default=str(RACINE / "data" / "ground_truth"))
    p.add_argument("--sortie", default=str(RACINE / "results" / "etape3"))
    p.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    p.add_argument("--debruitage", default="nlm")
    p.add_argument("--contraste", default="clahe")
    p.add_argument("--fenetre", type=int, default=51, help="fenêtre des méthodes locales (px)")
    p.add_argument("--polarite", default="auto", choices=["auto", "fonce", "clair"])
    p.add_argument("--sans-nettoyage", action="store_true")
    p.add_argument("--ocr", action="store_true", help="mesurer CER/WER Tesseract")
    p.add_argument("--balayage", action="store_true", help="balayage fenêtre × k pour Sauvola (avec --ocr)")
    p.add_argument("--langue", default="fra+eng")
    p.add_argument("--psm", default="11", help="mode de segmentation Tesseract (11 = texte épars)")
    args = p.parse_args()

    images = lister_images(args.images)
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        sys.exit(f"Aucune image trouvée dans {args.images}")

    sortie = Path(args.sortie)
    (sortie / "figures").mkdir(parents=True, exist_ok=True)
    if args.ocr:
        import pytesseract
        config = f"--psm {args.psm}"

    locales = {"fenetre": args.fenetre}
    nettoyage = not args.sans_nettoyage
    print(f"Prétraitement : {args.debruitage} + {args.contraste} | fenêtre locale : {args.fenetre} px"
          f" | nettoyage : {'oui' if nettoyage else 'non'}")

    f_csv = open(sortie / "binarisation.csv", "w", newline="", encoding="utf-8")
    w_csv = csv.writer(f_csv)
    w_csv.writerow(["image", "inversee", "methode", "temps_ms", "taux_pixels_texte",
                    "nb_composantes", *METRIQUES, "nb_mots_lus", "texte_ocr"])
    if args.balayage:
        f_bal = open(sortie / "balayage_sauvola.csv", "w", newline="", encoding="utf-8")
        w_bal = csv.writer(f_bal)
        w_bal.writerow(["image", "fenetre", "k", "cer", "f1_mots"])

    scores = defaultdict(lambda: defaultdict(list))  # méthode -> métrique -> valeurs
    temps_par_methode = defaultdict(list)
    f1_balayage = defaultdict(list)

    for n, chemin in enumerate(images, 1):
        nom = chemin.stem
        _, gris = acquerir(chemin)
        pre = pretraiter(gris, args.debruitage, args.contraste)
        gris_norm, inversee = normaliser_polarite(pre, args.polarite)
        print(f"[{n}/{len(images)}] {chemin.name}  (polarité : texte {'clair → inversé' if inversee else 'foncé'})")

        gt = Path(args.verite) / f"{nom}.txt"
        reference = gt.read_text(encoding="utf-8").strip() if gt.is_file() else ""
        faire_ocr = args.ocr and bool(reference)
        if args.ocr and not reference:
            print("     (pas de vérité terrain remplie : OCR ignoré)")

        # Référence : Tesseract binarise lui-même l'image en gris
        resultats = {}
        if faire_ocr:
            texte = pytesseract.image_to_string(gris_norm, lang=args.langue, config=config)
            sc = evaluer(reference, texte)
            for m in METRIQUES:
                scores["gris (Tesseract)"][m].append(sc[m])
            w_csv.writerow([chemin.name, inversee, "gris (Tesseract)", "", "", "",
                            *(f"{sc[m]:.3f}" for m in METRIQUES), len(texte.split()),
                            " ".join(texte.split())])

        for methode, fonction in BINARISATION.items():
            params = {} if methode in GLOBALES else locales
            t0 = time.perf_counter()
            b = fonction(gris_norm, **params)
            if nettoyage:
                b = nettoyer(b)
            dt = (time.perf_counter() - t0) * 1000
            resultats[methode] = b
            temps_par_methode[methode].append(dt)

            nb_comp = cv2.connectedComponents((b == 0).astype(np.uint8))[0] - 1
            taux = float((b == 0).mean())
            texte = ""
            valeurs = [""] * len(METRIQUES)
            if faire_ocr:
                texte = pytesseract.image_to_string(b, lang=args.langue, config=config)
                sc = evaluer(reference, texte)
                for m in METRIQUES:
                    scores[methode][m].append(sc[m])
                valeurs = [f"{sc[m]:.3f}" for m in METRIQUES]
            w_csv.writerow([chemin.name, inversee, methode, f"{dt:.1f}", f"{taux:.4f}",
                            nb_comp, *valeurs, len(texte.split()) if faire_ocr else "",
                            " ".join(texte.split())])

        # Figures
        titres = ["prétraitée"] + list(resultats)
        imgs = [gris_norm] + list(resultats.values())
        grille_methodes(imgs, titres, sortie / "figures" / f"{nom}_binarisation.png",
                        colonnes=4, histogrammes=False,
                        titre_general=f"{chemin.name} — binarisations (fenêtre locale {args.fenetre} px)")
        zoom_comparatif(imgs, titres, zone_la_plus_texturee(gris_norm),
                        sortie / "figures" / f"{nom}_zoom.png")
        seuils = {"fixe": 127,
                  "otsu": cv2.threshold(gris_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0],
                  "triangle": cv2.threshold(gris_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]}
        histogramme_seuils(gris_norm, seuils, sortie / "figures" / f"{nom}_histogramme.png",
                           titre=f"{chemin.name} — seuils globaux")

        # Balayage des paramètres de Sauvola
        if args.balayage and faire_ocr:
            for f in FENETRES:
                for k in VALEURS_K:
                    b = sauvola(gris_norm, fenetre=f, k=k)
                    if nettoyage:
                        b = nettoyer(b)
                    texte = pytesseract.image_to_string(b, lang=args.langue, config=config)
                    sc = evaluer(reference, texte)
                    f1_balayage[(f, k)].append(sc["f1_mots"])
                    w_bal.writerow([chemin.name, f, k, f"{sc['cer']:.3f}", f"{sc['f1_mots']:.3f}"])

    f_csv.close()
    if args.balayage:
        f_bal.close()

    # --- Synthèse ---------------------------------------------------------
    temps = {m: float(np.mean(v)) for m, v in temps_par_methode.items()}
    print("\n=== Temps moyen de binarisation (ms) ===")
    print("  " + "  ".join(f"{m}={v:.1f}" for m, v in temps.items()))
    barres_metriques(temps, "Temps moyen de binarisation", "ms", sortie / "synthese_temps.png")

    if scores:
        # Mêmes métriques que la baseline, en médianes (voir baseline_tesseract.py)
        nb = len(next(iter(scores.values()))["cer"])
        ordre = sorted(scores, key=lambda m: -np.median(scores[m]["f1_mots"]))
        print(f"\n=== OCR selon la binarisation ({nb} image(s) annotée(s), médianes) ===")
        print(f"  {'méthode':20s} {'CER':>7s} {'WER':>7s} {'précision':>10s} {'rappel':>7s} {'F1':>6s}")
        for m in ordre:
            v = scores[m]
            print(f"  {m:20s} {np.median(v['cer']):7.3f} {np.median(v['wer']):7.3f} "
                  f"{np.median(v['precision_mots']):10.3f} {np.median(v['rappel_mots']):7.3f} "
                  f"{np.median(v['f1_mots']):6.3f}")
        with open(sortie / "synthese_ocr.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["methode", "n"] + [f"{m}_{s}" for m in METRIQUES for s in ("moy", "med")])
            for m in ordre:
                w.writerow([m, nb] + [f"{fn(scores[m][x]):.3f}" for x in METRIQUES
                                      for fn in (np.mean, np.median)])
        for x, titre in (("cer", "CER médian"), ("rappel_mots", "Rappel médian des mots"),
                         ("f1_mots", "F1 médian des mots")):
            barres_metriques({m: float(np.median(scores[m][x])) for m in ordre},
                             f"{titre} selon la binarisation", x, sortie / f"synthese_{x}.png")
    elif args.ocr:
        print("\nAucune image avec vérité terrain remplie : pas de score OCR calculé.")

    if f1_balayage:
        print("\n=== Sauvola : F1 médian des mots selon la fenêtre (lignes) et k (colonnes) ===")
        print("  fenêtre " + "".join(f"{k:>8}" for k in VALEURS_K))
        for f in FENETRES:
            print(f"  {f:7d} " + "".join(f"{np.median(f1_balayage[(f, k)]):8.3f}" for k in VALEURS_K))

    print(f"\nRésultats : {sortie}")


if __name__ == "__main__":
    main()

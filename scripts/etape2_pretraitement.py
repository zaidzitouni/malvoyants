#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etape2_pretraitement.py — Comparaison des méthodes de débruitage et de contraste.

Pour chaque image de data/images/ :
  A. Débruitage SANS référence : on applique les 5 méthodes à la photo réelle,
     figure + zoom sur la zone la plus texturée (souvent le texte).
  B. Débruitage AVEC référence : on considère la photo comme « propre », on
     ajoute un bruit connu (gaussien σ=15, poivre et sel 5 %), puis on mesure
     PSNR / SSIM / temps de chaque méthode.
  C. Contraste : après débruitage NLM, on compare 5 méthodes de contraste
     (figure + histogrammes + contraste RMS, entropie, netteté).
  D. (option --ocr) Impact direct sur Tesseract : CER de l'OCR sur l'image
     entière pour chaque combinaison, si data/ground_truth/<nom>.txt existe.

Sorties dans results/etape2/ :
  figures/<image>_debruitage.png, _zoom.png, _contraste.png
  debruitage_reference.csv, contraste.csv, [ocr_pretraitement.csv, synthese_ocr.csv]
  synthese_*.png (moyennes sur toute la base)

Usage (depuis la racine du projet) :
  python scripts/etape2_pretraitement.py
  python scripts/etape2_pretraitement.py --max-images 5 --ocr
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
from metrics_image import (ajouter_bruit, contraste_rms, entropie,  # noqa: E402
                           nettete, psnr, ssim)
from evaluation import evaluer  # noqa: E402
from preprocessing import CONTRASTE, DEBRUITAGE, estimer_bruit, pretraiter  # noqa: E402

METRIQUES = ("cer", "wer", "precision_mots", "rappel_mots", "f1_mots")
from visualization import barres_metriques, grille_methodes, zoom_comparatif  # noqa: E402

BRUITS = [("gaussien", 15), ("poivre_sel", 5)]


def zone_la_plus_texturee(gris, taille=160):
    """Fenêtre carrée où l'énergie du Laplacien est maximale : sur une photo
    de plaque ou de panneau, c'est presque toujours une zone de texte."""
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
    p.add_argument("--sortie", default=str(RACINE / "results" / "etape2"))
    p.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    p.add_argument("--ocr", action="store_true", help="mesurer le CER Tesseract (lent)")
    p.add_argument("--langue", default="fra+eng")
    p.add_argument("--psm", default="11", help="mode Tesseract (11 : meilleur en baseline)")
    p.add_argument("--seuil-reference", type=float, default=6.0,
                   help="σ estimé max pour qu'une image serve de référence propre (partie B)")
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

    f_deb = open(sortie / "debruitage_reference.csv", "w", newline="", encoding="utf-8")
    f_con = open(sortie / "contraste.csv", "w", newline="", encoding="utf-8")
    w_deb, w_con = csv.writer(f_deb), csv.writer(f_con)
    w_deb.writerow(["image", "bruit", "methode", "psnr_db", "ssim", "temps_ms"])
    w_con.writerow(["image", "bruit_estime", "methode", "contraste_rms", "entropie_bits", "nettete"])
    if args.ocr:
        f_ocr = open(sortie / "ocr_pretraitement.csv", "w", newline="", encoding="utf-8")
        w_ocr = csv.writer(f_ocr)
        w_ocr.writerow(["image", "debruitage", "contraste", *METRIQUES, "nb_mots_lus", "texte_ocr"])

    moy_ssim = defaultdict(list)
    moy_temps = defaultdict(list)
    moy_rms = defaultdict(list)
    scores_ocr = defaultdict(lambda: defaultdict(list))  # combinaison -> métrique -> valeurs

    for k, chemin in enumerate(images, 1):
        nom = chemin.stem
        print(f"[{k}/{len(images)}] {chemin.name}")
        _, gris = acquerir(chemin)
        sigma = estimer_bruit(gris)
        print(f"     bruit estimé σ ≈ {sigma:.1f}")

        # --- A. Débruitage sur la photo réelle ---------------------------------
        sorties = {m: f(gris) for m, f in DEBRUITAGE.items()}
        grille_methodes(list(sorties.values()), list(sorties.keys()),
                        sortie / "figures" / f"{nom}_debruitage.png",
                        colonnes=5, histogrammes=False,
                        titre_general=f"{chemin.name} — débruitage (σ estimé ≈ {sigma:.1f})")
        zoom_comparatif(list(sorties.values()), list(sorties.keys()),
                        zone_la_plus_texturee(gris), sortie / "figures" / f"{nom}_zoom.png")

        # --- B. Débruitage avec référence ---------------------------------------
        # La photo sert de « vérité » : elle doit donc être déjà propre. Sinon
        # un bon filtre retire AUSSI son bruit d'origine et s'écarte de la
        # référence bruitée : il serait pénalisé à tort.
        if sigma > args.seuil_reference:
            print(f"     σ > {args.seuil_reference} : image trop bruitée pour servir de référence (B ignoré)")
            bruits_image = []
        else:
            bruits_image = BRUITS
        for type_bruit, intensite in bruits_image:
            bruitee = ajouter_bruit(gris, type_bruit, intensite)
            etiquette = f"{type_bruit}_{intensite}"
            for m, f in DEBRUITAGE.items():
                t0 = time.perf_counter()
                res = f(bruitee)
                dt = (time.perf_counter() - t0) * 1000
                s = ssim(gris, res)
                w_deb.writerow([chemin.name, etiquette, m, f"{psnr(gris, res):.2f}", f"{s:.4f}", f"{dt:.1f}"])
                moy_ssim[(etiquette, m)].append(s)
                moy_temps[m].append(dt)

        # --- C. Contraste (après NLM) -------------------------------------------
        base = DEBRUITAGE["nlm"](gris)
        versions = {m: f(base) for m, f in CONTRASTE.items()}
        for m, img in versions.items():
            r = contraste_rms(img)
            w_con.writerow([chemin.name, f"{sigma:.2f}", m, f"{r:.4f}",
                            f"{entropie(img):.3f}", f"{nettete(img):.1f}"])
            moy_rms[m].append(r)
        grille_methodes(list(versions.values()), [f"NLM + {m}" for m in versions],
                        sortie / "figures" / f"{nom}_contraste.png", colonnes=5,
                        titre_general=f"{chemin.name} — contraste (histogrammes dessous)")

        # --- D. Impact OCR ------------------------------------------------------
        gt = Path(args.verite) / f"{nom}.txt"
        reference = gt.read_text(encoding="utf-8").strip() if gt.is_file() else ""
        if args.ocr and reference:
            for d in ("aucun", "median", "bilateral", "nlm"):
                for c in ("aucun", "egalisation", "clahe"):
                    img = pretraiter(gris, d, c)
                    texte = pytesseract.image_to_string(img, lang=args.langue, config=f"--psm {args.psm}")
                    sc = evaluer(reference, texte)
                    w_ocr.writerow([chemin.name, d, c, *(f"{sc[m]:.3f}" for m in METRIQUES),
                                    len(texte.split()), " ".join(texte.split())])
                    for m in METRIQUES:
                        scores_ocr[f"{d}+{c}"][m].append(sc[m])
        elif args.ocr:
            print(f"     (pas de vérité terrain remplie pour {gt.name} : OCR ignoré)")

    f_deb.close()
    f_con.close()
    if args.ocr:
        f_ocr.close()

    # --- Synthèse sur toute la base -------------------------------------------
    print("\n=== SSIM moyen après débruitage (1 = parfait) ===")
    for type_bruit, intensite in BRUITS:
        etiquette = f"{type_bruit}_{intensite}"
        if not moy_ssim[(etiquette, "aucun")]:
            print("  aucune image assez propre pour servir de référence")
            break
        valeurs = {m: float(np.mean(moy_ssim[(etiquette, m)])) for m in DEBRUITAGE}
        print(f"  {etiquette:14s} " + "  ".join(f"{m}={v:.3f}" for m, v in valeurs.items()))
        barres_metriques(valeurs, f"SSIM moyen — bruit {etiquette}", "SSIM",
                         sortie / f"synthese_ssim_{etiquette}.png")

    if moy_temps:
        temps = {m: float(np.mean(v)) for m, v in moy_temps.items()}
        print("\n=== Temps moyen (ms) ===\n  " + "  ".join(f"{m}={v:.0f}" for m, v in temps.items()))
        barres_metriques(temps, "Temps moyen de débruitage", "ms", sortie / "synthese_temps.png")

    rms = {m: float(np.mean(v)) for m, v in moy_rms.items()}
    print("\n=== Contraste RMS moyen ===\n  " + "  ".join(f"{m}={v:.3f}" for m, v in rms.items()))
    barres_metriques(rms, "Contraste RMS moyen (après NLM)", "RMS", sortie / "synthese_contraste.png")

    if scores_ocr:
        # Mêmes métriques que la baseline (scripts/baseline_tesseract.py) pour
        # une comparaison directe. Médianes : le CER peut dépasser 1 et
        # quelques images « explosent » la moyenne.
        n = len(next(iter(scores_ocr.values()))["cer"])
        ordre = sorted(scores_ocr, key=lambda k: -np.median(scores_ocr[k]["f1_mots"]))
        print(f"\n=== OCR selon le prétraitement ({n} image(s), psm {args.psm}, médianes) ===")
        print(f"  {'combinaison':22s} {'CER':>7s} {'WER':>7s} {'précision':>10s} {'rappel':>7s} {'F1':>6s}")
        for k in ordre:
            v = scores_ocr[k]
            print(f"  {k:22s} {np.median(v['cer']):7.3f} {np.median(v['wer']):7.3f} "
                  f"{np.median(v['precision_mots']):10.3f} {np.median(v['rappel_mots']):7.3f} "
                  f"{np.median(v['f1_mots']):6.3f}")
        with open(sortie / "synthese_ocr.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["combinaison", "n"] + [f"{m}_{s}" for m in METRIQUES for s in ("moy", "med")])
            for k in ordre:
                w.writerow([k, n] + [f"{fn(scores_ocr[k][m]):.3f}" for m in METRIQUES
                                     for fn in (np.mean, np.median)])
        for m, titre in (("cer", "CER médian"), ("rappel_mots", "Rappel médian des mots"),
                         ("f1_mots", "F1 médian des mots")):
            barres_metriques({k: float(np.median(scores_ocr[k][m])) for k in ordre},
                             f"{titre} selon le prétraitement", m, sortie / f"synthese_{m}.png")

    print(f"\nRésultats : {sortie}")


if __name__ == "__main__":
    main()

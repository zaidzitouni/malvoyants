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
  debruitage_reference.csv, contraste.csv, [ocr_pretraitement.csv]
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
from preprocessing import CONTRASTE, DEBRUITAGE, estimer_bruit, pretraiter  # noqa: E402
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


def cer(reference, hypothese):
    """Character Error Rate = distance de Levenshtein / longueur de la référence.
    (Version simple ici ; le module evaluation.py complet viendra à l'étape 7.)"""
    r = " ".join(reference.lower().split())
    h = " ".join(hypothese.lower().split())
    if not r:
        return 0.0 if not h else 1.0
    prec = list(range(len(h) + 1))
    for i, cr in enumerate(r, 1):
        cour = [i]
        for j, ch in enumerate(h, 1):
            cour.append(min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (cr != ch)))
        prec = cour
    return prec[-1] / len(r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(RACINE / "data" / "images"))
    p.add_argument("--verite", default=str(RACINE / "data" / "ground_truth"))
    p.add_argument("--sortie", default=str(RACINE / "results" / "etape2"))
    p.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    p.add_argument("--ocr", action="store_true", help="mesurer le CER Tesseract (lent)")
    p.add_argument("--langue", default="fra+eng")
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
        w_ocr.writerow(["image", "debruitage", "contraste", "cer", "texte_ocr"])

    moy_ssim = defaultdict(list)
    moy_temps = defaultdict(list)
    moy_rms = defaultdict(list)
    moy_cer = defaultdict(list)

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
        if args.ocr and gt.is_file():
            reference = gt.read_text(encoding="utf-8")
            for d in ("aucun", "median", "bilateral", "nlm"):
                for c in ("aucun", "egalisation", "clahe"):
                    img = pretraiter(gris, d, c)
                    texte = pytesseract.image_to_string(img, lang=args.langue, config="--psm 11")
                    e = cer(reference, texte)
                    w_ocr.writerow([chemin.name, d, c, f"{e:.3f}", " ".join(texte.split())])
                    moy_cer[f"{d}+{c}"].append(e)
        elif args.ocr:
            print(f"     (pas de vérité terrain {gt.name} : OCR ignoré)")

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

    if moy_cer:
        # Le CER peut dépasser 1 quand Tesseract « hallucine » des caractères
        # parasites dans le fond : on affiche aussi la médiane, moins sensible.
        cers = {k: float(np.mean(v)) for k, v in sorted(moy_cer.items(), key=lambda kv: np.mean(kv[1]))}
        print("\n=== CER Tesseract (plus bas = mieux) ===")
        print(f"  {'combinaison':22s} {'moyen':>7s} {'médian':>7s}")
        for k, v in cers.items():
            print(f"  {k:22s} {v:7.3f} {float(np.median(moy_cer[k])):7.3f}")
        barres_metriques(cers, "CER moyen Tesseract selon le prétraitement", "CER",
                         sortie / "synthese_cer.png")

    print(f"\nRésultats : {sortie}")


if __name__ == "__main__":
    main()

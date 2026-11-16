#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etape4_localisation.py — Localisation du texte + OCR par région.

Compare, avec les MÊMES métriques que la baseline (CER, WER, précision,
rappel, F1 des mots ; médianes) :

  sans_localisation   Tesseract psm 11 sur l'image brute entière
                      (= la meilleure configuration de la baseline)
  <détecteur>+<binarisation>
      détecteur    : morpho, mser, union (les deux)
      binarisation : aucune (Tesseract binarise lui-même la région),
                     otsu, sauvola — appliquée à CHAQUE région séparément

Pour chaque image :
  figures/<image>_detection.png   boîtes trouvées par morpho et par MSER
                                  (rouge fin : candidats rejetés par la
                                  vérification ; vert : régions retenues)
  figures/<image>_regions.png     régions telles que Tesseract les reçoit
                                  (configuration --config-planches), avec le
                                  texte lu et la confiance
  textes/<image>.txt              texte restitué (même configuration)

Sorties globales : localisation.csv, synthese_localisation.csv, synthese_*.png

Usage (depuis la racine du projet) :
  uv run scripts/etape4_localisation.py --max-images 5
  uv run scripts/etape4_localisation.py
  uv run scripts/etape4_localisation.py --detecteurs morpho --binarisations otsu sauvola
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

from acquisition import acquerir, charger_image, lister_images  # noqa: E402
from evaluation import evaluer  # noqa: E402
from ocr import lire_regions  # noqa: E402
from text_detection import detecter_texte  # noqa: E402
from visualization import barres_metriques  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIQUES = ("cer", "wer", "precision_mots", "rappel_mots", "f1_mots")


def dessiner_detections(couleur, resultats, chemin, titre):
    """Une vignette par détecteur : candidats (rouge) et retenues (vert)."""
    fig, axes = plt.subplots(1, len(resultats), figsize=(6 * len(resultats), 6 * couleur.shape[0] / couleur.shape[1] + 0.6))
    for ax, (nom, (boites, infos)) in zip(np.atleast_1d(axes), resultats.items()):
        img = cv2.cvtColor(couleur, cv2.COLOR_BGR2RGB).copy()
        ep = max(2, img.shape[1] // 400)
        for x, y, w, h in infos.get("candidats", []):
            cv2.rectangle(img, (x, y), (x + w, y + h), (230, 40, 40), max(1, ep // 2))
        for x, y, w, h in boites:
            cv2.rectangle(img, (x, y), (x + w, y + h), (20, 200, 60), ep * 2)
        ax.imshow(img)
        ax.set_title(f"{nom} : {len(boites)} région(s) retenue(s) / {len(infos.get('candidats', []))} candidats",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def planche_regions(details, chemin, titre, maximum=24):
    """Les régions binarisées envoyées à Tesseract, avec le texte lu."""
    details = details[:maximum]
    if not details:
        return
    fig, axes = plt.subplots(len(details), 1, figsize=(6, 0.9 * len(details) + 0.5), squeeze=False)
    for ax, d in zip(axes[:, 0], details):
        ax.imshow(d["image"], cmap="gray", vmin=0, vmax=255)
        lu = d["texte"] or "(rien de fiable)"
        ax.set_title(f"{lu}   [confiance {d['confiance']:.0f}]", fontsize=8, loc="left")
        ax.axis("off")
    fig.suptitle(titre, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (0.9 * len(details) + 0.5)))
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(RACINE / "data" / "images"))
    p.add_argument("--verite", default=str(RACINE / "data" / "ground_truth"))
    p.add_argument("--sortie", default=str(RACINE / "results" / "etape4"))
    p.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    p.add_argument("--detecteurs", nargs="+", default=["morpho", "mser", "union"])
    p.add_argument("--binarisations", nargs="+", default=["aucune", "otsu", "sauvola"])
    p.add_argument("--config-planches", default="morpho+otsu",
                   help="configuration utilisée pour les planches de régions et les textes restitués")
    p.add_argument("--confiance", type=float, default=50, help="confiance Tesseract minimale par mot")
    p.add_argument("--langue", default="fra+eng")
    p.add_argument("--sans-reference", action="store_true",
                   help="ne pas recalculer « sans_localisation » (gain de temps)")
    args = p.parse_args()

    import pytesseract

    paires = []
    for chemin in lister_images(args.images):
        gt = Path(args.verite) / f"{chemin.stem}.txt"
        if gt.is_file() and gt.read_text(encoding="utf-8").strip():
            paires.append((chemin, gt.read_text(encoding="utf-8")))
    if args.max_images:
        paires = paires[:args.max_images]
    if not paires:
        sys.exit("Aucune image avec vérité terrain remplie.")

    sortie = Path(args.sortie)
    for d in ("figures", "textes"):
        (sortie / d).mkdir(parents=True, exist_ok=True)
    configs = [f"{d}+{b}" for d in args.detecteurs for b in args.binarisations]
    print(f"{len(paires)} image(s) | configurations : {', '.join(configs)}"
          f"{'' if args.sans_reference else ' + sans_localisation'}")

    scores = defaultdict(lambda: defaultdict(list))
    temps = defaultdict(list)
    nb_regions = defaultdict(list)

    with open(sortie / "localisation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "configuration", "nb_regions", *METRIQUES, "temps_s", "nb_mots_lus", "texte_ocr"])
        for n, (chemin, reference) in enumerate(paires, 1):
            couleur, gris = acquerir(chemin)
            resume = []

            if not args.sans_reference:
                t0 = time.perf_counter()
                brute = cv2.cvtColor(charger_image(chemin), cv2.COLOR_BGR2RGB)
                texte = pytesseract.image_to_string(brute, lang=args.langue, config="--psm 11")
                dt = time.perf_counter() - t0
                sc = evaluer(reference, texte)
                for m in METRIQUES:
                    scores["sans_localisation"][m].append(sc[m])
                temps["sans_localisation"].append(dt)
                w.writerow([chemin.name, "sans_localisation", "", *(f"{sc[m]:.3f}" for m in METRIQUES),
                            f"{dt:.2f}", len(texte.split()), " ".join(texte.split())])
                resume.append(f"sans_loc F1={sc['f1_mots']:.2f}")

            detections = {}
            for det in args.detecteurs:
                t0 = time.perf_counter()
                detections[det] = (*detecter_texte(gris, det), time.perf_counter() - t0)

            for det in args.detecteurs:
                boites, infos, t_det = detections[det]
                for binz in args.binarisations:
                    cfg = f"{det}+{binz}"
                    t0 = time.perf_counter()
                    texte, details = lire_regions(gris, boites, binz, args.langue, confiance_min=args.confiance)
                    dt = t_det + time.perf_counter() - t0
                    sc = evaluer(reference, texte)
                    for m in METRIQUES:
                        scores[cfg][m].append(sc[m])
                    temps[cfg].append(dt)
                    nb_regions[cfg].append(len(boites))
                    w.writerow([chemin.name, cfg, len(boites), *(f"{sc[m]:.3f}" for m in METRIQUES),
                                f"{dt:.2f}", len(texte.split()), " | ".join(texte.splitlines())])
                    if cfg == args.config_planches:
                        planche_regions(details, sortie / "figures" / f"{chemin.stem}_regions.png",
                                        f"{chemin.name} — régions envoyées à Tesseract ({cfg})")
                        (sortie / "textes" / f"{chemin.stem}.txt").write_text(texte + "\n", encoding="utf-8")
                        resume.append(f"{cfg} F1={sc['f1_mots']:.2f}")

            dessiner_detections(couleur, {d: detections[d][:2] for d in args.detecteurs if d != "union"} or
                                {d: detections[d][:2] for d in args.detecteurs},
                                sortie / "figures" / f"{chemin.stem}_detection.png", chemin.name)
            print(f"[{n}/{len(paires)}] {chemin.name:22s} " + "  ".join(resume))

    # --- Synthèse ---------------------------------------------------------
    ordre = sorted(scores, key=lambda k: -np.median(scores[k]["f1_mots"]))
    nb = len(paires)
    print(f"\n=== LOCALISATION + OCR PAR RÉGION ({nb} image(s), médianes) ===")
    print(f"  {'configuration':20s} {'CER':>7s} {'WER':>7s} {'précision':>10s} {'rappel':>7s} "
          f"{'F1':>6s} {'régions':>8s} {'temps':>7s}")
    for k in ordre:
        v = scores[k]
        reg = f"{np.median(nb_regions[k]):8.0f}" if nb_regions[k] else f"{'—':>8s}"
        print(f"  {k:20s} {np.median(v['cer']):7.3f} {np.median(v['wer']):7.3f} "
              f"{np.median(v['precision_mots']):10.3f} {np.median(v['rappel_mots']):7.3f} "
              f"{np.median(v['f1_mots']):6.3f} {reg} {np.mean(temps[k]):6.1f}s")

    with open(sortie / "synthese_localisation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "n"] + [f"{m}_{s}" for m in METRIQUES for s in ("moy", "med")]
                   + ["regions_med", "temps_moy_s"])
        for k in ordre:
            w.writerow([k, nb] + [f"{fn(scores[k][m]):.3f}" for m in METRIQUES for fn in (np.mean, np.median)]
                       + [f"{np.median(nb_regions[k]):.0f}" if nb_regions[k] else "",
                          f"{np.mean(temps[k]):.2f}"])

    for m, titre in (("f1_mots", "F1 médian des mots"), ("precision_mots", "Précision médiane des mots"),
                     ("rappel_mots", "Rappel médian des mots"), ("cer", "CER médian")):
        barres_metriques({k: float(np.median(scores[k][m])) for k in ordre},
                         f"Localisation — {titre}", m, sortie / f"synthese_{m}.png")
    print(f"\nRésultats : {sortie}")


if __name__ == "__main__":
    main()

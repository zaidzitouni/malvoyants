#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py — Étape 7 : comparaison finale de toutes les méthodes.

Rassemble les résultats déjà calculés par les autres scripts (aucun OCR n'est
relancé ici) :
  results/baseline/baseline.csv            Tesseract sur l'image brute
  results/etape2/ocr_pretraitement.csv     débruitage × contraste (image entière)
  results/etape3/binarisation.csv          binarisations (image entière)
  results/etape4*/localisation.csv         localisation + OCR par région
                                           (chaque dossier etape4* = une variante)
Les fichiers absents sont simplement ignorés.

Statistiques (sur les images communes à toutes les configurations comparées) :
  - médiane et moyenne de F1, précision, rappel, CER ;
  - INTERVALLE DE CONFIANCE À 95 % de la médiane du F1, par bootstrap
    (2 000 tirages avec remise des images) : avec 50 images, une différence
    de médiane qui reste dans les intervalles n'est pas démontrée ;
  - comparaison APPARIÉE à la baseline (même image) : nombre d'images
    améliorées / égales / dégradées et test des rangs signés de Wilcoxon
    (non paramétrique : les F1 ne suivent pas une loi normale, beaucoup
    valent 0).

Sorties dans results/benchmark/ (reprises telles quelles par le rapport et
la présentation LaTeX, qui se mettent à jour en recompilant) :
  benchmark_complet.csv     une ligne par configuration
  tableau_principal.tex     progression baseline → configuration finale
  tableau_etape4.tex        toutes les variantes de localisation
  tableau_categories.tex    baseline vs finale, par catégorie d'image
  chiffres.tex              macros LaTeX des chiffres cités dans le texte
  progression.pdf / .png    F1 médian + IC 95 % par étape
  precision_rappel.pdf/.png précision vs rappel des configurations

Usage :
  uv run scripts/benchmark.py
  uv run scripts/benchmark.py --finale etape4_v2:mser+otsu
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
METRIQUES = ("cer", "wer", "precision_mots", "rappel_mots", "f1_mots")
REFERENCE = "baseline:psm 11"


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def _lire(chemin):
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def charger(resultats):
    """Renvoie {configuration: {image: {métrique: valeur}}} et {configuration: étape}."""
    donnees = defaultdict(dict)
    etape = {}

    def ajouter(cfg, et, ligne):
        if not ligne.get("f1_mots"):
            return
        donnees[cfg][ligne["image"]] = {m: float(ligne[m]) for m in METRIQUES}
        etape[cfg] = et

    f = resultats / "baseline" / "baseline.csv"
    if f.is_file():
        for l in _lire(f):
            ajouter(f"baseline:psm {l['psm']}", "1. baseline", l)
    f = resultats / "etape2" / "ocr_pretraitement.csv"
    if f.is_file() and "f1_mots" in (_lire(f) or [{}])[0]:
        for l in _lire(f):
            ajouter(f"etape2:{l['debruitage']}+{l['contraste']}", "2. prétraitement", l)
    f = resultats / "etape3" / "binarisation.csv"
    if f.is_file() and "f1_mots" in (_lire(f) or [{}])[0]:
        for l in _lire(f):
            ajouter(f"etape3:{l['methode']}", "3. binarisation", l)
    for d in sorted(resultats.glob("etape4*")):
        f = d / "localisation.csv"
        if f.is_file():
            for l in _lire(f):
                if l["configuration"] == "sans_localisation":
                    continue
                ajouter(f"{d.name}:{l['configuration']}", "4. localisation", l)
    return donnees, etape


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def ic_mediane(valeurs, n_tirages=2000, graine=0):
    rng = np.random.default_rng(graine)
    v = np.asarray(valeurs)
    tirages = rng.choice(v, size=(n_tirages, v.size), replace=True)
    meds = np.median(tirages, axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def resumer(donnees, etape, images):
    lignes = []
    ref = donnees.get(REFERENCE)
    for cfg, par_image in donnees.items():
        if not all(i in par_image for i in images):
            continue
        tab = {m: np.array([par_image[i][m] for i in images]) for m in METRIQUES}
        bas, haut = ic_mediane(tab["f1_mots"])
        r = {"configuration": cfg, "etape": etape[cfg], "n": len(images),
             "f1_med": float(np.median(tab["f1_mots"])), "f1_ic_bas": bas, "f1_ic_haut": haut,
             "f1_moy": float(tab["f1_mots"].mean()),
             "precision_med": float(np.median(tab["precision_mots"])),
             "precision_moy": float(tab["precision_mots"].mean()),
             "rappel_med": float(np.median(tab["rappel_mots"])),
             "rappel_moy": float(tab["rappel_mots"].mean()),
             "cer_med": float(np.median(tab["cer"])), "wer_med": float(np.median(tab["wer"]))}
        if ref and cfg != REFERENCE:
            a = np.array([ref[i]["f1_mots"] for i in images])
            d = tab["f1_mots"] - a
            r["mieux"], r["egal"], r["pire"] = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
            r["p_wilcoxon"] = float(wilcoxon(tab["f1_mots"], a, zero_method="zsplit").pvalue) if np.any(d) else 1.0
        lignes.append(r)
    return sorted(lignes, key=lambda r: (r["etape"], -r["f1_med"], -r["f1_moy"]))


# ---------------------------------------------------------------------------
# Mise en forme LaTeX (virgule décimale française)
# ---------------------------------------------------------------------------

def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", "{,}")


def fr_p(p):
    if p < 0.001:
        return r"$<$\,0{,}001"
    return fr(p, 3)


def nom_tex(cfg):
    return cfg.replace("_", r"\_").replace("+", " + ").replace(":", " : ")


def tableau(lignes, chemin, legende, etiquette, avec_test=True):
    entete = r"Configuration & F1 méd. [IC 95\,\%] & F1 moy. & Préc. méd. & Rappel méd. & CER méd."
    colonnes = "lccccc"
    if avec_test:
        entete += r" & $+/=/-$ & $p$"
        colonnes += "cc"
    corps = []
    for r in lignes:
        ligne = (f"{nom_tex(r['configuration'])} & {fr(r['f1_med'])} [{fr(r['f1_ic_bas'])}\\,; {fr(r['f1_ic_haut'])}]"
                 f" & {fr(r['f1_moy'])} & {fr(r['precision_med'])} & {fr(r['rappel_med'])} & {fr(r['cer_med'])}")
        if avec_test:
            if "mieux" in r:
                ligne += f" & {r['mieux']}/{r['egal']}/{r['pire']} & {fr_p(r['p_wilcoxon'])}"
            else:
                ligne += " & \\multicolumn{2}{c}{référence}"
        corps.append(ligne + r" \\")
    tex = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colonnes}}}", r"\toprule", entete + r" \\", r"\midrule",
        *corps, r"\bottomrule", r"\end{tabular}}",
        rf"\caption{{{legende}}}", rf"\label{{{etiquette}}}", r"\end{table}", ""])
    Path(chemin).write_text(tex, encoding="utf-8")


def tableau_categories(donnees, images, finale, chemin):
    cats = sorted({i.split("_")[0] for i in images})
    corps = []
    for c in cats + ["toutes"]:
        ii = images if c == "toutes" else [i for i in images if i.split("_")[0] == c]
        b = [donnees[REFERENCE][i] for i in ii]
        f = [donnees[finale][i] for i in ii]
        med = lambda L, m: float(np.median([x[m] for x in L]))
        nom = r"\textbf{toutes}" if c == "toutes" else c
        corps.append(f"{nom} & {len(ii)} & {fr(med(b, 'f1_mots'))} & {fr(med(f, 'f1_mots'))} & "
                     f"{fr(med(b, 'precision_mots'))} & {fr(med(f, 'precision_mots'))} & "
                     f"{fr(med(b, 'rappel_mots'))} & {fr(med(f, 'rappel_mots'))} \\\\")
        if c == cats[-1]:
            corps.append(r"\midrule")
    tex = "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{lrcccccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{F1 médian} & \multicolumn{2}{c}{Précision méd.} & \multicolumn{2}{c}{Rappel méd.} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Catégorie & $n$ & base & finale & base & finale & base & finale \\", r"\midrule",
        *corps, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Baseline (Tesseract psm 11 sur l'image brute) et configuration finale ({nom_tex(finale)}), par catégorie d'image.}}",
        r"\label{tab:categories}", r"\end{table}", ""])
    Path(chemin).write_text(tex, encoding="utf-8")


def macros(resume, finale, donnees, images, chemin):
    par_cfg = {r["configuration"]: r for r in resume}
    b, f = par_cfg[REFERENCE], par_cfg[finale]
    base3 = par_cfg.get("baseline:psm 3")
    mots = lambda cfg: None
    lignes = {
        "nbImages": str(len(images)),
        "configFinale": nom_tex(finale),
        "baseFmed": fr(b["f1_med"], 3), "baseFmoy": fr(b["f1_moy"], 3),
        "basePmed": fr(b["precision_med"], 3), "baseRmed": fr(b["rappel_med"], 3),
        "baseCERmed": fr(b["cer_med"], 2),
        "finFmed": fr(f["f1_med"], 3), "finFmoy": fr(f["f1_moy"], 3),
        "finFicBas": fr(f["f1_ic_bas"], 2), "finFicHaut": fr(f["f1_ic_haut"], 2),
        "finPmed": fr(f["precision_med"], 3), "finRmed": fr(f["rappel_med"], 3),
        "finCERmed": fr(f["cer_med"], 2),
        "finMieux": str(f.get("mieux", "")), "finEgal": str(f.get("egal", "")), "finPire": str(f.get("pire", "")),
        "finP": fr_p(f.get("p_wilcoxon", 1.0)),
        "facteurF": fr(f["f1_med"] / b["f1_med"], 0) if b["f1_med"] > 0 else "--",
    }
    if base3:
        lignes["basePsmTroisFmed"] = fr(base3["f1_med"], 3)
    tex = "% Généré par scripts/benchmark.py — ne pas modifier à la main\n" + "\n".join(
        rf"\newcommand{{\{k}}}{{{v}}}" for k, v in lignes.items()) + "\n"
    Path(chemin).write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COULEURS = {"1. baseline": "#8C8C8C", "2. prétraitement": "#4C72B0", "3. binarisation": "#55A868",
            "4. localisation": "#C44E52"}


def _virgule(ax):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    ax.yaxis.set_major_formatter(f)
    ax.xaxis.set_major_formatter(f) if ax.get_xlabel() else None


def figure_progression(cles, chemin_base):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    noms = [r["configuration"].split(":", 1)[1] if ":" in r["configuration"] else r["configuration"] for r in cles]
    x = np.arange(len(cles))
    for i, r in enumerate(cles):
        ax.bar(i, r["f1_med"], color=COULEURS.get(r["etape"], "#666"), width=0.65)
        ax.errorbar(i, r["f1_med"], yerr=[[r["f1_med"] - r["f1_ic_bas"]], [r["f1_ic_haut"] - r["f1_med"]]],
                    color="black", capsize=4, lw=1)
        ax.text(i, r["f1_ic_haut"] + 0.01, f"{r['f1_med']:.2f}".replace(".", ","), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['etape'].split('. ')[1]}\n{n}" for r, n in zip(cles, noms)], fontsize=7.5)
    ax.set_ylabel("F1 médian des mots")
    ax.set_title("F1 médian par étape (barres : IC à 95 % par bootstrap)", fontsize=10)
    _virgule(ax)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{chemin_base}.{ext}", dpi=160)
    plt.close(fig)


def figure_precision_rappel(resume, chemin_base):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    vues = set()
    for r in resume:
        e = r["etape"]
        ax.scatter(r["rappel_moy"], r["precision_moy"], s=40, color=COULEURS.get(e, "#666"),
                   label=e if e not in vues else None, edgecolor="white", lw=0.5)
        vues.add(e)
    decalage = {"mser+aucune": (4, -10)}
    for r in resume:
        if r["etape"] in ("1. baseline",) or r["configuration"].endswith(("mser+otsu", "mser+aucune", "morpho+otsu")):
            nom = r["configuration"].split(":", 1)[1]
            ax.annotate(nom, (r["rappel_moy"], r["precision_moy"]), fontsize=7,
                        xytext=decalage.get(nom, (4, 3)), textcoords="offset points")
    ax.set_xlabel("Rappel moyen des mots")
    ax.set_ylabel("Précision moyenne des mots")
    ax.set_xlim(0, max(0.5, max(r["rappel_moy"] for r in resume) + 0.05))
    ax.set_ylim(0, max(0.5, max(r["precision_moy"] for r in resume) + 0.05))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Compromis précision / rappel", fontsize=10)
    _virgule(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{chemin_base}.{ext}", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resultats", default=str(RACINE / "results"))
    p.add_argument("--sortie", default=str(RACINE / "results" / "benchmark"))
    p.add_argument("--finale", default="etape4:mser+otsu",
                   help="configuration finale, sous la forme <dossier>:<configuration>")
    args = p.parse_args()

    resultats, sortie = Path(args.resultats), Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    donnees, etape = charger(resultats)
    if REFERENCE not in donnees:
        sys.exit("Baseline introuvable : lancez d'abord scripts/baseline_tesseract.py")
    if args.finale not in donnees:
        sys.exit(f"Configuration finale introuvable : {args.finale}\nDisponibles : "
                 + ", ".join(k for k in donnees if k.startswith("etape4")))

    # Images communes à la baseline et à la configuration finale
    images = sorted(set(donnees[REFERENCE]) & set(donnees[args.finale]))
    resume = resumer(donnees, etape, images)
    ignorees = [c for c in donnees if c not in {r["configuration"] for r in resume}]
    print(f"{len(images)} images communes | {len(resume)} configurations comparées")
    if ignorees:
        print(f"  (ignorées, pas calculées sur toutes ces images : {', '.join(ignorees)})")

    with open(sortie / "benchmark_complet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resume[0].keys()) + ["mieux", "egal", "pire", "p_wilcoxon"],
                           extrasaction="ignore")
        w.writeheader()
        for r in resume:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Progression : baseline psm 3 et 11, meilleure config des étapes 2 et 3, finale
    par_etape = defaultdict(list)
    for r in resume:
        par_etape[r["etape"]].append(r)
    par_cfg = {r["configuration"]: r for r in resume}
    cles = sorted(par_etape["1. baseline"], key=lambda r: int(r["configuration"].split()[-1]))
    for e in ("2. prétraitement", "3. binarisation"):
        if par_etape[e]:
            cles.append(max(par_etape[e], key=lambda r: (r["f1_med"], r["f1_moy"])))
    cles.append(par_cfg[args.finale])
    tableau(cles, sortie / "tableau_principal.tex",
            f"Progression de la baseline à la configuration finale ({len(images)} images). "
            r"Pour les étapes 2 et 3, meilleure configuration de l'étape. "
            r"$+/=/-$ : images où le F1 est meilleur, égal ou moins bon que la baseline psm 11 ; "
            r"$p$ : test apparié des rangs signés de Wilcoxon.", "tab:principal")
    tableau(par_etape["4. localisation"], sortie / "tableau_etape4.tex",
            f"Variantes de localisation et de binarisation par région ({len(images)} images).", "tab:etape4")
    for e, fichier in (("2. prétraitement", "tableau_etape2.tex"), ("3. binarisation", "tableau_etape3.tex")):
        if par_etape[e]:
            tableau(par_etape[e], sortie / fichier, f"Résultats de l'étape « {e[3:]} » ({len(images)} images).",
                    f"tab:{fichier[8:14]}")
    tableau_categories(donnees, images, args.finale, sortie / "tableau_categories.tex")
    macros(resume, args.finale, donnees, images, sortie / "chiffres.tex")
    figure_progression(cles, sortie / "progression")
    figure_precision_rappel(resume, sortie / "precision_rappel")
    (sortie / "resume.json").write_text(json.dumps({"images": images, "finale": args.finale,
                                                    "configurations": resume}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")

    print(f"\n{'configuration':32s} {'F1 méd [IC95]':>20s} {'F1 moy':>7s} {'P méd':>6s} {'R méd':>6s} {'+/=/-':>9s} {'p':>7s}")
    for r in resume:
        t = f"{r['mieux']}/{r['egal']}/{r['pire']}" if "mieux" in r else "réf."
        pv = f"{r['p_wilcoxon']:.3f}" if "p_wilcoxon" in r else ""
        print(f"{r['configuration']:32s} {r['f1_med']:5.3f} [{r['f1_ic_bas']:.2f};{r['f1_ic_haut']:.2f}]"
              f" {r['f1_moy']:7.3f} {r['precision_med']:6.3f} {r['rappel_med']:6.3f} {t:>9s} {pv:>7s}")
    print(f"\nSorties : {sortie}")


if __name__ == "__main__":
    main()

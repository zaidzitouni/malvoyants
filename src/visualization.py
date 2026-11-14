# -*- coding: utf-8 -*-
"""
visualization.py — Figures Matplotlib pour le rapport et la présentation.

Les fonctions enregistrent directement des PNG (dpi 150) sans ouvrir de
fenêtre (backend "Agg"), ce qui permet de lancer le benchmark sur toute la
base sans interaction.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _histogramme(ax, gris, titre=None):
    hist = np.bincount(gris.ravel(), minlength=256)
    ax.fill_between(np.arange(256), hist, color="#4C72B0", alpha=0.8, linewidth=0)
    cdf = hist.cumsum() / hist.sum() * hist.max()
    ax.plot(cdf, color="#C44E52", linewidth=1, label="cumulé (normalisé)")
    ax.set_xlim(0, 255)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    if titre:
        ax.set_title(titre, fontsize=8)


def grille_methodes(images, titres, chemin, colonnes=3, histogrammes=True, titre_general=None):
    """Grille image (+ histogramme dessous) pour comparer plusieurs méthodes."""
    n = len(images)
    lignes = int(np.ceil(n / colonnes))
    hauteurs = [3, 1] * lignes if histogrammes else [3] * lignes
    fig, axes = plt.subplots(
        len(hauteurs), colonnes, figsize=(4 * colonnes, sum(hauteurs) * 0.9 + 0.6),
        gridspec_kw={"height_ratios": hauteurs}, squeeze=False,
    )
    pas = 2 if histogrammes else 1
    for i in range(lignes * colonnes):
        l, c = divmod(i, colonnes)
        ax_img = axes[l * pas][c]
        ax_hist = axes[l * pas + 1][c] if histogrammes else None
        if i < n:
            ax_img.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax_img.set_title(titres[i], fontsize=9)
            if ax_hist is not None:
                _histogramme(ax_hist, images[i])
        else:
            if ax_hist is not None:
                ax_hist.axis("off")
        ax_img.axis("off")
    if titre_general:
        fig.suptitle(titre_general, fontsize=11)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def zoom_comparatif(images, titres, boite, chemin):
    """Même zone agrandie sur chaque version : montre l'effet sur les bords
    des lettres, invisible à l'échelle de l'image entière.
    boite = (x, y, largeur, hauteur)."""
    x, y, w, h = boite
    fig, axes = plt.subplots(1, len(images), figsize=(2.6 * len(images), 2.6 * h / w + 0.5))
    for ax, img, t in zip(np.atleast_1d(axes), images, titres):
        ax.imshow(img[y:y + h, x:x + w], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def barres_metriques(valeurs, titre, ylabel, chemin):
    """Diagramme en barres simple : {méthode: valeur}."""
    noms = list(valeurs)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(noms)), 3.4))
    ax.bar(noms, [valeurs[k] for k in noms], color="#4C72B0")
    if len(noms) > 5:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    vals = [v for v in valeurs.values() if v > 0]
    if vals and max(vals) / min(vals) > 50:
        ax.set_yscale("symlog", linthresh=1)  # écarts énormes : échelle log lisible
    ax.set_title(titre, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

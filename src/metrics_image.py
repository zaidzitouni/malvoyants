# -*- coding: utf-8 -*-
"""
metrics_image.py — Mesures objectives de qualité d'image.

Deux familles :

1. AVEC référence (image propre connue) — pour le débruitage, sur des images
   auxquelles on a AJOUTÉ un bruit contrôlé :
       PSNR  rapport signal/bruit crête, en dB (plus haut = mieux) ;
       SSIM  similarité structurelle dans [0, 1] (1 = identique), plus proche
             de la perception humaine que le PSNR car il compare luminance,
             contraste et structure locale.

2. SANS référence — pour le contraste, sur les vraies photos :
       contraste RMS   écart-type des intensités normalisées ;
       entropie        quantité d'information de l'histogramme (bits) ;
       netteté         variance du Laplacien (bords francs = valeur élevée).

Le vrai juge restera le CER de l'OCR (étape 7) : ces mesures servent à
expliquer POURQUOI une méthode aide ou nuit à l'OCR.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(reference, image):
    mse = np.mean((reference.astype(np.float64) - image.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(reference, image, sigma=1.5):
    """SSIM (Wang et al., 2004) avec fenêtre gaussienne, implémenté avec SciPy."""
    x = reference.astype(np.float64)
    y = image.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    vx = gaussian_filter(x * x, sigma) - mx ** 2
    vy = gaussian_filter(y * y, sigma) - my ** 2
    cxy = gaussian_filter(x * y, sigma) - mx * my
    carte = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(carte.mean())


def contraste_rms(gris):
    return float((gris.astype(np.float64) / 255.0).std())


def entropie(gris):
    hist = np.bincount(gris.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def nettete(gris):
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def ajouter_bruit(gris, type_bruit="gaussien", intensite=15, graine=0):
    """Bruit contrôlé pour évaluer le débruitage avec référence.
    gaussien      : intensite = écart-type σ ;
    poivre_sel    : intensite = pourcentage de pixels touchés."""
    rng = np.random.default_rng(graine)
    if type_bruit == "gaussien":
        bruit = rng.normal(0, intensite, gris.shape)
        return np.clip(gris.astype(np.float64) + bruit, 0, 255).astype(np.uint8)
    if type_bruit == "poivre_sel":
        sortie = gris.copy()
        tirage = rng.random(gris.shape)
        p = intensite / 100.0
        sortie[tirage < p / 2] = 0
        sortie[tirage > 1 - p / 2] = 255
        return sortie
    raise ValueError(f"Type de bruit inconnu : {type_bruit}")

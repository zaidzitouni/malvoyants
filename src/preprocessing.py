# -*- coding: utf-8 -*-
"""
preprocessing.py — Étape 2 du pipeline : débruitage et amélioration du contraste.

Toutes les fonctions prennent et renvoient une image en niveaux de gris uint8.
Elles sont regroupées dans deux registres (DEBRUITAGE, CONTRASTE) afin que le
pipeline et le benchmark puissent choisir une méthode par son nom, ce qui rend
la comparaison systématique (étape 7) triviale.

Choix des méthodes (résumé, détaillé dans le rapport) :

DÉBRUITAGE
    gaussien   Référence. Lisse le bruit mais floute aussi les bords des
               lettres : c'est le contre-exemple à montrer.
    median     Remplace chaque pixel par la médiane du voisinage. Idéal contre
               le bruit impulsionnel (poussière, pixels morts, « poivre et sel »)
               et préserve assez bien les contours.
    bilateral  Moyenne pondérée par la distance spatiale ET la ressemblance
               d'intensité : lisse les zones homogènes sans traverser les bords.
               Bon compromis pour les plaques à fort contraste.
    nlm        Non-Local Means : moyenne de patchs similaires dans toute une
               fenêtre de recherche. Meilleur sur le bruit gaussien des capteurs
               de smartphone (photos de nuit), mais 10 à 50× plus lent.

CONTRASTE
    egalisation  Égalisation d'histogramme globale : étale l'histogramme sur
                 [0, 255]. Simple, mais sur-amplifie le bruit et sature les zones
                 claires quand l'éclairage n'est pas uniforme.
    clahe        Égalisation adaptative par tuiles, avec limite de contraste
                 (clip limit) qui empêche l'amplification du bruit. Méthode
                 recommandée pour l'espace public (ombres, reflets).
    gamma        Correction non linéaire I' = 255·(I/255)^γ. γ < 1 éclaircit les
                 images sous-exposées ; γ choisi automatiquement si non fourni.
    etirement    Étirement linéaire entre les percentiles 2 % et 98 % : robuste
                 aux quelques pixels extrêmes (reflets).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Débruitage
# ---------------------------------------------------------------------------

def debruitage_gaussien(gris, ksize=5, sigma=0):
    """Flou gaussien. sigma=0 → calculé par OpenCV à partir de ksize."""
    return cv2.GaussianBlur(gris, (ksize, ksize), sigma)


def debruitage_median(gris, ksize=3):
    """Filtre médian. ksize impair ; 3 suffit pour du bruit léger, 5 pour du
    bruit impulsionnel dense (au-delà, les traits fins des lettres s'effacent)."""
    return cv2.medianBlur(gris, ksize)


def debruitage_bilateral(gris, d=9, sigma_couleur=50, sigma_espace=50):
    """Filtre bilatéral.
    d             diamètre du voisinage ;
    sigma_couleur écart d'intensité au-delà duquel deux pixels ne se mélangent
                  plus (≈ contraste minimal d'un bord à préserver) ;
    sigma_espace  portée spatiale du lissage."""
    return cv2.bilateralFilter(gris, d, sigma_couleur, sigma_espace)


def debruitage_nlm(gris, h=None, taille_patch=7, taille_recherche=21):
    """Non-Local Means.
    h : force du filtrage. Si None, on l'adapte au bruit estimé (h ≈ 1,2 σ̂),
    ce qui évite de sur-lisser une image déjà propre."""
    if h is None:
        h = float(np.clip(1.2 * estimer_bruit(gris), 3, 25))
    return cv2.fastNlMeansDenoising(gris, None, h, taille_patch, taille_recherche)


def estimer_bruit(gris):
    """Estimation de l'écart-type du bruit (méthode de Immerkær, 1996).

    On convolue avec un masque qui annule les zones lisses et les rampes,
    il ne reste (presque) que le bruit. Rapide et sans paramètre."""
    masque = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = gris.shape
    reponse = cv2.filter2D(gris.astype(np.float64), -1, masque)
    return float(np.sqrt(np.pi / 2) * np.abs(reponse).sum() / (6 * (w - 2) * (h - 2)))


# ---------------------------------------------------------------------------
# Contraste
# ---------------------------------------------------------------------------

def egalisation_globale(gris):
    """Égalisation d'histogramme classique."""
    return cv2.equalizeHist(gris)


def clahe(gris, clip_limit=2.0, tuiles=(8, 8)):
    """CLAHE.
    clip_limit : 2 à 3 est le bon intervalle pour du texte ; au-delà, le grain
                 du fond (crépi, tôle) devient aussi contrasté que les lettres.
    tuiles     : 8×8 sur une image de 1600 px → tuiles de 200 px, soit à peu
                 près la taille d'une plaque : chaque plaque est égalisée
                 localement, indépendamment de l'ombre qui l'entoure."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuiles).apply(gris)


def correction_gamma(gris, gamma=None):
    """Correction gamma via une table de correspondance (LUT, très rapide).
    Si gamma est None, on le choisit pour ramener la moyenne vers 128 :
    γ = log(0,5) / log(moyenne/255)."""
    if gamma is None:
        moyenne = max(float(gris.mean()), 1.0) / 255.0
        gamma = float(np.clip(np.log(0.5) / np.log(moyenne), 0.3, 3.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** gamma).clip(0, 255).astype(np.uint8)
    return cv2.LUT(gris, lut)


def etirement_percentiles(gris, bas=2, haut=98):
    """Étirement linéaire robuste entre deux percentiles."""
    p_bas, p_haut = np.percentile(gris, (bas, haut))
    if p_haut - p_bas < 1:
        return gris.copy()
    sortie = (gris.astype(np.float32) - p_bas) * 255.0 / (p_haut - p_bas)
    return sortie.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Registres et point d'entrée
# ---------------------------------------------------------------------------

DEBRUITAGE = {
    "aucun": lambda g: g.copy(),
    "gaussien": debruitage_gaussien,
    "median": debruitage_median,
    "bilateral": debruitage_bilateral,
    "nlm": debruitage_nlm,
}

CONTRASTE = {
    "aucun": lambda g: g.copy(),
    "egalisation": egalisation_globale,
    "clahe": clahe,
    "gamma": correction_gamma,
    "etirement": etirement_percentiles,
}


def pretraiter(gris, debruitage="nlm", contraste="clahe"):
    """Applique débruitage PUIS contraste.

    L'ordre compte : égaliser d'abord amplifierait le bruit, que le filtre
    devrait ensuite combattre avec plus de force (et donc plus de flou)."""
    if debruitage not in DEBRUITAGE:
        raise KeyError(f"Débruitage inconnu : {debruitage} ({list(DEBRUITAGE)})")
    if contraste not in CONTRASTE:
        raise KeyError(f"Contraste inconnu : {contraste} ({list(CONTRASTE)})")
    return CONTRASTE[contraste](DEBRUITAGE[debruitage](gris))

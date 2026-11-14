# -*- coding: utf-8 -*-
"""
acquisition.py — Étape 1 du pipeline : chargement et normalisation des images.

Rôle :
    - lire une image depuis le disque (chemins avec accents compris) ;
    - la convertir en niveaux de gris (l'OCR et la binarisation travaillent sur
      la luminance) tout en conservant la version couleur pour l'affichage ;
    - normaliser sa taille pour que les paramètres des étapes suivantes
      (tailles de noyaux, de tuiles CLAHE, d'éléments structurants) aient
      le même effet d'une image à l'autre.

Justification de la normalisation de taille :
    Une photo de smartphone fait 4000 px de large, une vignette Commons 1600 px.
    Un filtre médian 5×5 n'a pas du tout le même effet sur les deux. En ramenant
    le plus grand côté à une valeur fixe (1600 px par défaut), on rend les
    paramètres comparables et on réduit le temps de calcul (NLM surtout).
"""

from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def charger_image(chemin):
    """Charge une image en BGR (uint8).

    cv2.imread échoue silencieusement sur les chemins non ASCII sous Windows ;
    on passe donc par np.fromfile + cv2.imdecode, qui fonctionne partout.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    donnees = np.fromfile(str(chemin), dtype=np.uint8)
    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Format non lisible : {chemin}")
    return image


def en_niveaux_de_gris(image):
    """BGR → niveaux de gris (pondération standard de la luminance)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def redimensionner(image, cote_max=1600, agrandir=False):
    """Ramène le plus grand côté à `cote_max` en conservant le ratio.

    - Réduction : INTER_AREA (moyenne des pixels, évite le crénelage).
    - Agrandissement (option) : INTER_CUBIC, plus net pour les petits caractères.
    """
    h, w = image.shape[:2]
    echelle = cote_max / max(h, w)
    if echelle >= 1 and not agrandir:
        return image
    interp = cv2.INTER_AREA if echelle < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * echelle), round(h * echelle)), interpolation=interp)


def acquerir(chemin, cote_max=1600):
    """Point d'entrée de l'étape 1.

    Retourne (couleur, gris), les deux à la même taille normalisée.
    """
    couleur = redimensionner(charger_image(chemin), cote_max)
    return couleur, en_niveaux_de_gris(couleur)


def lister_images(dossier):
    """Liste récursive et triée des images d'un dossier."""
    return sorted(p for p in Path(dossier).rglob("*") if p.suffix.lower() in EXTENSIONS)

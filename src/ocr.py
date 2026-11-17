# -*- coding: utf-8 -*-
"""
ocr.py — Étape 5 du pipeline : reconnaissance du texte, région par région.

Pour chaque boîte renvoyée par la localisation (étape 4) :
  1. RECADRAGE avec une marge (le fond autour des lettres aide la
     binarisation locale et Tesseract, qui coupe sinon la 1re et la dernière
     lettre).
  2. MISE À L'ÉCHELLE : Tesseract reconnaît le mieux des caractères d'environ
     30 à 40 px de haut. On redimensionne la région pour que la ligne fasse
     HAUTEUR_CIBLE px (agrandissement bicubique pour les petits panneaux
     lointains, réduction pour les grandes enseignes).
  3. BINARISATION locale de la seule région (étape 3) : la polarité est
     estimée sur la région elle-même, là où elle est fiable, et la fenêtre de
     Sauvola est calculée à partir de la hauteur de la ligne.
  4. TESSERACT en mode « une ligne » (psm 7), adapté puisque chaque boîte est
     une ligne. image_to_data fournit une CONFIANCE par mot (0-100).
  5. FILTRAGE des mots : confiance minimale, et au moins un caractère
     alphanumérique (élimine « | », « — », « © » qui viennent des bords).

Le texte final est l'assemblage des lignes dans l'ordre de lecture.
"""

import cv2
import numpy as np
import pytesseract

from binarization import binariser, fenetre_auto
from text_detection import agrandir

HAUTEUR_CIBLE = 48          # hauteur de la ligne après redimensionnement (px)
CONFIANCE_MIN = 50          # en dessous, le mot est jugé non fiable


def preparer_region(gris, boite, binarisation="sauvola", hauteur_cible=HAUTEUR_CIBLE):
    """Recadre, redimensionne et binarise une région. Renvoie l'image prête
    pour Tesseract (texte noir sur fond blanc)."""
    x, y, w, h = agrandir(boite, gris.shape, marge=0.25)
    region = gris[y:y + h, x:x + w]
    echelle = hauteur_cible / max(1, boite[3])
    interp = cv2.INTER_CUBIC if echelle > 1 else cv2.INTER_AREA
    region = cv2.resize(region, None, fx=echelle, fy=echelle, interpolation=interp)
    if binarisation in (None, "aucune"):
        return region
    params = {}
    if binarisation in ("sauvola", "niblack", "adapt_moyenne", "adapt_gauss"):
        params["fenetre"] = fenetre_auto(hauteur_cible)
    b = nettoyer_region(binariser(region, binarisation, nettoyage=False, **params))
    # Liseré blanc : Tesseract lit mal le texte qui touche le bord de l'image
    return cv2.copyMakeBorder(b, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def nettoyer_region(binaire):
    """Retire d'une région recadrée ce qui n'est pas une lettre :
    - les objets qui TOUCHENT LE BORD de la région (cadre du panneau, flèche,
      poteau, morceau de panneau voisin) : grâce à la marge ajoutée au
      recadrage, une vraie lettre ne touche jamais le bord ;
    - les objets creux et étendus (cadre complet, remplissage < 15 %).
    Sans cela, Tesseract voit le texte « dans une case » et ne renvoie rien
    (constaté sur GANNAT et BELLERIVE-CENTRE, panneaux_006)."""
    texte = (binaire == 0).astype(np.uint8)
    H, W = texte.shape
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(texte, connectivity=8)
    garder = np.ones(n, dtype=bool)
    garder[0] = False
    for i in range(1, n):
        x, y, w, h, aire = stats[i]
        touche_bord = x == 0 or y == 0 or x + w >= W or y + h >= H
        creux = (w > 0.5 * W or h > 0.9 * H) and aire < 0.15 * w * h
        if touche_bord or creux:
            garder[i] = False
    return np.where(garder[etiq], 0, 255).astype(np.uint8)


def _mot_valide(mot):
    return any(c.isalnum() for c in mot)


def lire_region(image_prete, langue="fra+eng", psm=7, confiance_min=CONFIANCE_MIN):
    """OCR d'une région préparée. Renvoie (texte, confiance moyenne, mots)."""
    d = pytesseract.image_to_data(image_prete, lang=langue, config=f"--psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    mots = []
    for mot, conf in zip(d["text"], d["conf"]):
        mot = mot.strip()
        conf = float(conf)
        if mot and conf >= confiance_min and _mot_valide(mot):
            mots.append((mot, conf))
    texte = " ".join(m for m, _ in mots)
    conf_moy = float(np.mean([c for _, c in mots])) if mots else 0.0
    return texte, conf_moy, mots


def separabilite(region):
    """Qualité de la séparation texte / fond d'une région : part de la variance
    expliquée par le seuil d'Otsu (critère η d'Otsu, entre 0 et 1). Proche de 1
    quand la région a deux populations nettes (lettres / fond)."""
    if region.size == 0 or region.std() < 1:
        return 0.0
    t, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r = region.astype(np.float64)
    bas, haut = r[r <= t], r[r > t]
    if bas.size == 0 or haut.size == 0:
        return 0.0
    w0, w1 = bas.size / r.size, haut.size / r.size
    return float(w0 * w1 * (bas.mean() - haut.mean()) ** 2 / r.var())


def choisir_canal(images, boite):
    """Parmi plusieurs canaux (gris, b de Lab…), celui où la région est la mieux
    séparée de son fond : c'est lui qu'on donnera à Tesseract."""
    x, y, w, h = boite
    return max(images.values(), key=lambda im: separabilite(im[y:y + h, x:x + w]))


def lire_regions(gris, boites, binarisation="sauvola", langue="fra+eng", psm=7,
                 confiance_min=CONFIANCE_MIN, canaux=None):
    """OCR de toutes les régions (dans l'ordre fourni, normalement l'ordre de
    lecture). Renvoie (texte complet sur plusieurs lignes, détail par région).
    canaux : canaux supplémentaires (voir text_detection.canaux_couleur) ; pour
    chaque région on lit le canal où elle est la plus contrastée."""
    images = {"gris": gris, **(canaux or {})}
    details = []
    for boite in boites:
        source = choisir_canal(images, boite) if len(images) > 1 else gris
        img = preparer_region(source, boite, binarisation)
        texte, conf, _ = lire_region(img, langue, psm, confiance_min)
        details.append({"boite": boite, "texte": texte, "confiance": conf, "image": img})
    lignes = [d["texte"] for d in details if d["texte"]]
    return "\n".join(lignes), details

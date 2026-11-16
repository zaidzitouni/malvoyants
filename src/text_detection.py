# -*- coding: utf-8 -*-
"""
text_detection.py — Étape 4 du pipeline : localisation des zones de texte.

Pourquoi cette étape est décisive (constat de la baseline) : sur une photo de
rue entière, Tesseract lit 17 fois plus de « mots » qu'il n'y en a, en voyant
des lettres dans les arbres, les façades, les pavés. Les seules images bien
lues sont celles où la plaque remplit le cadre. On cherche donc les zones de
texte AVANT l'OCR, pour ne présenter à Tesseract que ces zones.

Entrée  : image en niveaux de gris uint8 (sortie de l'acquisition, ≤ 1600 px).
Sortie  : liste de boîtes (x, y, largeur, hauteur), une par LIGNE de texte.

Deux détecteurs, puis une vérification commune :

1. MORPHOLOGIE (« contours + morphologie mathématique »)
   a. Gradient morphologique (dilatation − érosion, élément 3×3) : fort sur
      les bords des lettres. Plus robuste que Canny pour des traits épais.
   b. Binarisation d'Otsu du gradient → carte des pixels « bord ».
   c. FERMETURE avec un élément structurant HORIZONTAL (≈ 2 % de la largeur ×
      3 px) : relie les lettres voisines d'une même ligne, mais pas deux lignes
      superposées (l'élément ne fait que 3 px de haut).
   d. Ouverture 3×3 : supprime les filaments isolés.
   e. Contours externes → boîtes candidates.

2. MSER (Maximally Stable Extremal Regions)
   ATTENTION : le MSER d'OpenCV 5.0.0 ne détecte aucune région, même sur un
   texte parfaitement net (vérifié : 0 région contre 11 avec OpenCV 4.14).
   Le projet fixe donc opencv-python-headless < 5 dans pyproject.toml.
   Une lettre est une région de couleur homogène, nettement séparée de son
   fond : quand on fait varier le seuil, sa forme reste STABLE. MSER extrait
   exactement ces régions, quelle que soit la polarité (clair sur foncé ou
   l'inverse). On filtre par forme (une lettre n'est ni minuscule, ni géante,
   ni filiforme), puis on regroupe en lignes les régions de hauteur voisine,
   alignées et proches horizontalement.

3. VÉRIFICATION par segmentation des caractères (commune aux deux)
   Les deux détecteurs trouvent aussi du feuillage, des grilles, des fenêtres.
   Une vraie ligne de texte, une fois binarisée, se découpe en plusieurs
   composantes connexes (les caractères) qui ont :
     - une HAUTEUR comparable entre elles (même corps de police),
     - une hauteur occupant une bonne part de la boîte,
     - un alignement vertical (bases et sommets voisins).
   Le feuillage donne des composantes de toutes tailles, mal alignées.
   On garde la boîte si elle contient au moins MIN_CARACTERES composantes
   compatibles ; sinon on l'écarte.
"""

import cv2
import numpy as np

from binarization import normaliser_polarite

MIN_CARACTERES = 2


# ---------------------------------------------------------------------------
# Outils sur les boîtes
# ---------------------------------------------------------------------------

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _inclusion(a, b):
    """Part de la boîte a contenue dans b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / (aw * ah) if aw * ah else 0.0


def fusionner_boites(boites, seuil_iou=0.3, seuil_inclusion=0.8):
    """Supprime les doublons : boîte très recouverte par une autre, ou contenue
    dans une plus grande. On garde la plus grande (souvent la ligne entière)."""
    boites = sorted(boites, key=lambda b: b[2] * b[3], reverse=True)
    gardees = []
    for b in boites:
        if all(_iou(b, g) < seuil_iou and _inclusion(b, g) < seuil_inclusion for g in gardees):
            gardees.append(b)
    return gardees


def ordre_de_lecture(boites):
    """Trie de haut en bas, puis de gauche à droite pour les boîtes d'une même
    « rangée » (centres verticaux à moins d'une demi-hauteur)."""
    boites = sorted(boites, key=lambda b: b[1] + b[3] / 2)
    rangees, courante = [], []
    for b in boites:
        if courante:
            yc = np.mean([c[1] + c[3] / 2 for c in courante])
            hm = np.mean([c[3] for c in courante])
            if abs(b[1] + b[3] / 2 - yc) > hm / 2:
                rangees.append(courante)
                courante = []
        courante.append(b)
    if courante:
        rangees.append(courante)
    return [b for r in rangees for b in sorted(r, key=lambda b: b[0])]


def agrandir(boite, forme, marge=0.15):
    """Ajoute une marge autour de la boîte (le fond autour des lettres aide la
    binarisation locale et Tesseract), sans sortir de l'image."""
    x, y, w, h = boite
    H, W = forme[:2]
    m = int(marge * h)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# 3. Vérification par segmentation des caractères
# ---------------------------------------------------------------------------

def segmenter_caracteres(region):
    """Binarise une région (polarité automatique, Otsu) et renvoie les boîtes
    des composantes qui ont une taille de caractère.
    Retourne (binaire texte=255, liste de (x, y, w, h))."""
    gris, _ = normaliser_polarite(region)
    _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    H, W = region.shape[:2]
    caracteres = []
    for i in range(1, n):
        x, y, w, h, aire = stats[i]
        if h < 0.3 * H or h > 0.98 * H:      # trop petit (bruit) / toute la hauteur (bord)
            continue
        if w > 1.5 * h or w > 0.9 * W:        # trop large pour un caractère
            continue
        if aire < 0.1 * w * h:                # presque vide : trait parasite
            continue
        caracteres.append((x, y, w, h))
    return b, caracteres


def verifier_ligne(gris, boite, min_caracteres=MIN_CARACTERES):
    """True si la boîte ressemble à une ligne de texte (voir en-tête, point 3).
    Renvoie aussi le nombre de caractères retenus (utile pour classer)."""
    x, y, w, h = agrandir(boite, gris.shape, marge=0.1)
    region = gris[y:y + h, x:x + w]
    if region.size == 0 or region.std() < 12:      # zone uniforme : pas de texte
        return False, 0
    _, car = segmenter_caracteres(region)
    if len(car) < min_caracteres:
        return False, len(car)
    hauteurs = np.array([c[3] for c in car], dtype=float)
    bases = np.array([c[1] + c[3] for c in car], dtype=float)
    # Garder le groupe majoritaire de hauteurs proches de la médiane
    med = np.median(hauteurs)
    proches = np.abs(hauteurs - med) < 0.3 * med
    if proches.sum() < min_caracteres:
        return False, int(proches.sum())
    # Alignement : bases des caractères proches (écart-type < 25 % de la hauteur)
    if np.std(bases[proches]) > 0.25 * med:
        return False, int(proches.sum())
    return True, int(proches.sum())


def decouper_lignes(gris, boite):
    """Découpe une zone de plusieurs lignes (« RANDAN / BRUGHEAS ») en lignes
    séparées, par PROFIL DE PROJECTION horizontal : on compte, rangée par
    rangée, les pixels des caractères ; les rangées vides séparent les lignes.
    Seules les composantes de taille caractère comptent : le cadre du panneau,
    qui touche toutes les rangées, est ignoré."""
    x, y, w, h = boite
    region = gris[y:y + h, x:x + w]
    if region.size == 0:
        return []
    gris_r, _ = normaliser_polarite(region)
    _, b = cv2.threshold(gris_r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    masque = np.zeros(b.shape, np.uint8)
    for i in range(1, n):
        cx, cy, cw, ch, aire = stats[i]
        if 0.08 * h <= ch <= 0.7 * h and cw <= 1.5 * ch and aire >= 0.1 * cw * ch:
            masque[etiq == i] = 1
    profil = masque.sum(axis=1)
    occupe = profil > 0.02 * max(1, profil.max())
    lignes, debut = [], None
    for r, o in enumerate(list(occupe) + [False]):
        if o and debut is None:
            debut = r
        elif not o and debut is not None:
            if r - debut >= 0.12 * h:
                lignes.append((debut, r))
            debut = None
    if len(lignes) < 2:
        return []
    sous = []
    for y0, y1 in lignes:
        colonnes = np.where(masque[y0:y1].any(axis=0))[0]
        if colonnes.size:
            sous.append((x + int(colonnes[0]), y + y0, int(colonnes[-1] - colonnes[0] + 1), y1 - y0))
    return sous


# ---------------------------------------------------------------------------
# 1. Détecteur morphologique
# ---------------------------------------------------------------------------

def candidats_morphologie(gris, largeur_fermeture=None):
    H, W = gris.shape
    lisse = cv2.bilateralFilter(gris, 7, 40, 40)
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(lisse, cv2.MORPH_GRADIENT, noyau)
    _, bords = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Retirer les longs traits horizontaux (bords de panneaux, corniches) :
    # une ouverture par un segment de 4 % de la largeur ne garde QUE ces
    # traits ; on les soustrait. Sans cela, la fermeture soude les lettres au
    # cadre du panneau et la boîte obtenue (cadre + texte) est rejetée.
    segment = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.04 * W)), 1))
    traits = cv2.morphologyEx(bords, cv2.MORPH_OPEN, segment)
    bords = cv2.subtract(bords, cv2.dilate(traits, np.ones((3, 3), np.uint8)))
    if largeur_fermeture is None:
        largeur_fermeture = max(9, int(0.02 * W))
    fermeture = cv2.getStructuringElement(cv2.MORPH_RECT, (largeur_fermeture, 3))
    lignes = cv2.morphologyEx(bords, cv2.MORPH_CLOSE, fermeture)
    lignes = cv2.morphologyEx(lignes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(lignes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boites = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (0.008 * H <= h <= 0.2 * H):         # hauteur de ligne plausible
            continue
        if w < 0.8 * h or w > 0.95 * W:              # « 42 » a w ≈ h ; pas toute la largeur
            continue
        remplissage = cv2.countNonZero(lignes[y:y + h, x:x + w]) / (w * h)
        if remplissage < 0.25:                       # forme très irrégulière (feuillage)
            continue
        boites.append((x, y, w, h))
    return boites, {"gradient": gradient, "bords": bords, "lignes": lignes}


# ---------------------------------------------------------------------------
# 2. Détecteur MSER
# ---------------------------------------------------------------------------

def _regrouper_en_lignes(lettres):
    """Regroupement glouton : deux lettres sont liées si leurs hauteurs sont
    proches (rapport < 1,5), leurs centres verticaux alignés (écart < 0,5 h)
    et l'écart horizontal < 1,2 h. Union-find sur ces liens."""
    n = len(lettres)
    parent = list(range(n))

    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ordre = sorted(range(n), key=lambda i: lettres[i][0])
    for a_idx, i in enumerate(ordre):
        xi, yi, wi, hi = lettres[i]
        for j in ordre[a_idx + 1:]:
            xj, yj, wj, hj = lettres[j]
            if xj - (xi + wi) > 1.2 * max(hi, hj):
                break                                  # trié par x : plus loin encore
            if max(hi, hj) / min(hi, hj) > 1.5:
                continue
            if abs((yi + hi / 2) - (yj + hj / 2)) > 0.5 * min(hi, hj):
                continue
            parent[racine(i)] = racine(j)
    groupes = {}
    for i in range(n):
        groupes.setdefault(racine(i), []).append(lettres[i])
    boites = []
    for g in groupes.values():
        if len(g) < MIN_CARACTERES:
            continue
        x0 = min(b[0] for b in g)
        y0 = min(b[1] for b in g)
        x1 = max(b[0] + b[2] for b in g)
        y1 = max(b[1] + b[3] for b in g)
        boites.append((x0, y0, x1 - x0, y1 - y0))
    return boites


def candidats_mser(gris, delta=5):
    H, W = gris.shape
    aire_min = max(30, int(1e-5 * H * W))
    aire_max = int(0.01 * H * W)
    mser = cv2.MSER_create(delta, aire_min, aire_max)
    # Lissage préalable : sur une photo bruitée, le bruit rend les régions
    # INSTABLES et MSER n'en garde presque aucune (1 seule région sur l'image
    # de test bruitée, contre ~70 après lissage).
    _, rects = mser.detectRegions(cv2.bilateralFilter(gris, 7, 40, 40))
    lettres = []
    for x, y, w, h in rects:
        if not (0.008 * H <= h <= 0.2 * H):
            continue
        if not (0.1 <= w / h <= 1.5):                # « I » fin … « W » large
            continue
        lettres.append((int(x), int(y), int(w), int(h)))
    lettres = fusionner_boites(lettres, seuil_iou=0.7, seuil_inclusion=0.95)
    return _regrouper_en_lignes(lettres), {"lettres": lettres}


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

DETECTEURS = {
    "morpho": candidats_morphologie,
    "mser": candidats_mser,
}


ECHELLES = (1.0, 0.5, 0.25)


def _candidats_multi_echelles(gris, detecteur, echelles=ECHELLES):
    """PYRAMIDE D'ÉCHELLES. Les noyaux des détecteurs ont une taille fixe en
    pixels (fermeture de 3 px de haut, gradient 3×3) : ils conviennent à des
    lettres de ~10 à 30 px. Une grande plaque photographiée de près (lettres
    de 70 px) se fragmente alors en bandes horizontales. On relance donc la
    détection sur l'image réduite de moitié puis au quart, où ces grandes
    lettres retrouvent une taille « normale », et on ramène les boîtes à
    l'échelle d'origine. Les doublons entre échelles sont fusionnés ensuite."""
    H, W = gris.shape
    candidats, infos = [], {}
    for e in echelles:
        if min(H, W) * e < 200:
            continue
        img = gris if e == 1.0 else cv2.resize(gris, None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
        boites, inf = detecteur(img)
        if e == 1.0:
            infos = inf
        candidats += [(int(x / e), int(y / e), int(w / e), int(h / e)) for x, y, w, h in boites]
    return candidats, infos


def detecter_texte(gris, methode="morpho", verifier=True, echelles=ECHELLES):
    """Renvoie (boîtes retenues dans l'ordre de lecture, infos de débogage).

    methode  : "morpho", "mser" ou "union" (les deux détecteurs combinés :
               plus de rappel, au prix de doublons fusionnés ensuite).
    echelles : pyramide utilisée ; (1.0,) pour désactiver."""
    if methode == "union":
        b1, i1 = _candidats_multi_echelles(gris, candidats_morphologie, echelles)
        b2, i2 = _candidats_multi_echelles(gris, candidats_mser, echelles)
        candidats, infos = b1 + b2, {**i1, **i2}
    elif methode in DETECTEURS:
        candidats, infos = _candidats_multi_echelles(gris, DETECTEURS[methode], echelles)
    else:
        raise KeyError(f"Détecteur inconnu : {methode} ({list(DETECTEURS) + ['union']})")
    infos["candidats"] = list(candidats)
    if verifier:
        retenus = []
        for b in candidats:
            if verifier_ligne(gris, b)[0]:
                retenus.append(b)
            else:  # peut-être plusieurs lignes collées : on tente de les séparer
                retenus += [s for s in decouper_lignes(gris, b) if verifier_ligne(gris, s)[0]]
        candidats = retenus
    boites = fusionner_boites(candidats)
    return ordre_de_lecture(boites), infos

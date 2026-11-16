# -*- coding: utf-8 -*-
"""
binarization.py — Étape 3 du pipeline : binarisation globale et adaptative.

Entrée  : image en niveaux de gris uint8 (sortie du prétraitement).
Sortie  : image binaire uint8 {0, 255} avec la convention attendue par
          Tesseract : TEXTE NOIR (0) SUR FOND BLANC (255).

Principe commun : un pixel devient « texte » si son intensité est inférieure
à un seuil T. Toute la différence entre les méthodes est dans le calcul de T.

MÉTHODES GLOBALES — un seul seuil T pour toute l'image
    fixe       T = 127. Référence naïve : ne s'adapte à rien.
    otsu       T qui maximise la variance inter-classes (texte / fond) de
               l'histogramme. Optimal si l'histogramme est bimodal, c'est-à-dire
               éclairage uniforme et deux populations nettes (plaque nette au
               soleil). Échoue dès qu'une ombre crée un 3e mode.
    triangle   T placé au « pied » du pic principal de l'histogramme. Adapté
               quand le texte est très minoritaire (petit numéro sur grand mur),
               cas où l'histogramme n'est pas bimodal et où Otsu se trompe.

MÉTHODES ADAPTATIVES (LOCALES) — un seuil T(x, y) par pixel, calculé sur une
fenêtre w×w centrée sur le pixel. Indispensables en extérieur : ombre portée,
dégradé de lumière, reflet sur une plaque émaillée.
    moyenne    T = moyenne locale − C            (OpenCV, ADAPTIVE_THRESH_MEAN_C)
    gaussienne T = moyenne locale pondérée − C   (ADAPTIVE_THRESH_GAUSSIAN_C) :
               les voisins proches comptent plus, bords un peu plus nets.
    niblack    T = m + k·s   (m, s : moyenne et écart-type locaux, k ≈ −0,2).
               Suit bien le texte mais transforme le bruit du fond en « texte »
               dans les zones uniformes (s faible → T ≈ m → tout bascule).
    sauvola    T = m · (1 + k·(s/R − 1)),  k = 0,5 ; R = 128 (valeurs de l'article
               original de Sauvola et Pietikäinen, 2000 ; k confirmé par le
               balayage de etape3_binarisation.py --balayage).
               Corrige Niblack : dans une zone uniforme (s petit), T descend
               nettement sous m et le fond reste blanc. Référence pour le texte
               imprimé, recommandée ici.

CHOIX DE LA FENÊTRE w (paramètre le plus important des méthodes locales)
    Elle doit couvrir un caractère ET un peu de fond autour, sinon au milieu
    d'un trait épais il n'y a que du « noir » et le seuil local coupe la lettre
    en deux. Règle pratique : w ≈ 1,5 à 2 × la hauteur d'un caractère. Pour
    nos images normalisées à 1600 px, le texte d'une plaque fait ~40-100 px
    → w = 51 par défaut. Sur une région de texte déjà localisée (étape 4), on
    calcule w à partir de la hauteur de la région (voir fenetre_auto).

POLARITÉ
    Les formules locales (Niblack, Sauvola, moyenne − C) supposent du texte
    FONCÉ sur fond CLAIR. Les plaques de rue marocaines/françaises sont souvent
    l'inverse (lettres blanches sur fond bleu). On détecte donc la polarité
    AVANT de seuiller et on inverse l'image de gris si besoin.
"""

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Polarité
# ---------------------------------------------------------------------------

def _zone_texturee(gris, taille):
    """Fenêtre carrée où l'énergie du Laplacien est maximale (≈ zone de texte)."""
    energie = np.abs(cv2.Laplacian(gris, cv2.CV_32F))
    somme = cv2.boxFilter(energie, -1, (taille, taille), normalize=False)
    m = taille // 2
    h, w = gris.shape
    interieur = somme[m:h - m or None, m:w - m or None]
    y, x = np.unravel_index(np.argmax(interieur), interieur.shape)
    return gris[y:y + taille, x:x + taille]


def _nb_objets_texte(masque, aire_min, aire_max):
    """Nombre de composantes connexes de taille « caractère »."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque.astype(np.uint8), connectivity=8)
    aires = stats[1:, cv2.CC_STAT_AREA]
    return int(((aires >= aire_min) & (aires <= aire_max)).sum())


def texte_clair_sur_fond_fonce(gris):
    """Devine si le texte est clair sur fond foncé.

    Idée : le texte, c'est la classe qui forme BEAUCOUP de petits objets séparés
    (une composante par lettre), alors que le fond forme peu de grandes plages.
    1. on se place sur la zone la plus texturée (la plaque, pas le mur) ;
       une région déjà recadrée (étape 4, petit côté ≤ 400 px) est analysée en entier ;
    2. on la binarise avec Otsu (sur une petite zone, l'ombre varie peu) ;
    3. on compte les objets de taille « caractère » dans chaque classe.

    Une première version regardait la couleur majoritaire sur les bords de
    l'image : elle se trompait sur les photos entières, où le bord est le mur
    et non le fond de la plaque."""
    h, w = gris.shape
    if min(h, w) > 400:  # photo entière : on cherche la plaque
        zone = _zone_texturee(gris, min(h, w) // 3)
    else:
        zone = gris
    _, b = cv2.threshold(zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    aire = zone.size
    aire_min, aire_max = max(4, int(5e-4 * aire)), int(0.15 * aire)
    objets_clairs = _nb_objets_texte(b == 255, aire_min, aire_max)
    objets_fonces = _nb_objets_texte(b == 0, aire_min, aire_max)
    return objets_clairs > objets_fonces


def normaliser_polarite(gris, polarite="auto"):
    """Renvoie (gris_corrigé, inversé?) avec du texte foncé sur fond clair.
    polarite : "auto", "fonce" (texte déjà foncé) ou "clair" (à inverser)."""
    inverser = texte_clair_sur_fond_fonce(gris) if polarite == "auto" else polarite == "clair"
    return (255 - gris if inverser else gris), inverser


# ---------------------------------------------------------------------------
# Méthodes globales
# ---------------------------------------------------------------------------

def seuil_fixe(gris, t=127):
    _, b = cv2.threshold(gris, t, 255, cv2.THRESH_BINARY)
    return b


def otsu(gris):
    _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def seuil_otsu_valeur(gris):
    """Valeur du seuil d'Otsu (pour l'afficher sur l'histogramme)."""
    t, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def triangle(gris):
    _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return b


# ---------------------------------------------------------------------------
# Méthodes adaptatives
# ---------------------------------------------------------------------------

def _impair(n):
    n = max(3, int(n))
    return n if n % 2 else n + 1


def fenetre_auto(hauteur_texte_px, facteur=1.8):
    """Fenêtre locale à partir de la hauteur (estimée) d'un caractère."""
    return _impair(facteur * hauteur_texte_px)


def adaptative_moyenne(gris, fenetre=51, c=10):
    """C : marge sous la moyenne. C > 0 évite que le grain d'un fond uniforme
    (où pixel ≈ moyenne) soit classé « texte »."""
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, _impair(fenetre), c)


def adaptative_gaussienne(gris, fenetre=51, c=10):
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, _impair(fenetre), c)


def _moyenne_ecart_type_locaux(gris, fenetre):
    """m(x,y) et s(x,y) sur une fenêtre carrée, via SciPy (uniform_filter).
    Coût indépendant de la taille de fenêtre (filtres séparables) :
    s² = E[I²] − E[I]²."""
    f = gris.astype(np.float64)
    m = uniform_filter(f, size=fenetre, mode="reflect")
    m2 = uniform_filter(f * f, size=fenetre, mode="reflect")
    s = np.sqrt(np.maximum(m2 - m * m, 0))
    return m, s


def niblack(gris, fenetre=51, k=-0.2):
    m, s = _moyenne_ecart_type_locaux(gris, _impair(fenetre))
    return np.where(gris > m + k * s, 255, 0).astype(np.uint8)


def sauvola(gris, fenetre=51, k=0.5, r=128.0):
    """k : sensibilité. Plus k est grand, plus le seuil descend sous la moyenne
    locale : le fond (mur, crépi) reste blanc, mais les traits très fins ou
    peu contrastés peuvent s'amincir. 0,2 garde ces traits fins mais laisse
    passer du grain du mur (lu comme du texte parasite par Tesseract).
    r : dynamique max de l'écart-type (128 pour des images 8 bits)."""
    m, s = _moyenne_ecart_type_locaux(gris, _impair(fenetre))
    t = m * (1.0 + k * (s / r - 1.0))
    return np.where(gris > t, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Post-traitement morphologique léger
# ---------------------------------------------------------------------------

def nettoyer(binaire, aire_min=None, ouverture=True, retirer_cadres=True):
    """Retire ce qui n'est pas du texte et gêne Tesseract.

    1. Ouverture 2×2 sur le texte (érosion puis dilatation) : supprime les
       points de 1 pixel sans changer l'épaisseur des traits.
    2. Composantes plus petites que aire_min supprimées (par défaut 0,002 %
       de l'image, ≈ 50 px pour 1600×1600 : bien plus petit qu'un point sur
       un i, bien plus grand qu'un grain de bruit).
    3. (retirer_cadres) Composantes qui ne peuvent pas être une lettre :
       - CADRES : grande boîte englobante (> 20 % de la largeur ou de la
         hauteur) mais presque vide (remplissage < 15 %). C'est le bord d'une
         plaque ou d'un panneau. Une lettre remplit 25 à 60 % de sa boîte.
       - LONGS TRAITS : rapport longueur/épaisseur > 15 et longueur > 20 %
         de l'image (soulignement, bord de panneau, poteau).
       Les méthodes adaptatives font ressortir ces bords ; Tesseract voit
       alors le texte « dans une case » et le classe comme tableau ou image :
       il ne renvoie rien (constaté sur les images de test)."""
    texte = (binaire == 0).astype(np.uint8)
    if ouverture:
        texte = cv2.morphologyEx(texte, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h_img, w_img = texte.shape
    if aire_min is None:
        aire_min = max(8, int(2e-5 * texte.size))
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(texte, connectivity=8)
    aire = stats[:, cv2.CC_STAT_AREA]
    garder = aire >= aire_min
    if retirer_cadres:
        w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
        h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        remplissage = aire / (w * h)
        grand = (w > 0.2 * w_img) | (h > 0.2 * h_img)
        cadre = grand & (remplissage < 0.15)
        allongement = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
        trait = (allongement > 15) & grand
        garder &= ~(cadre | trait)
    garder[0] = False  # étiquette 0 = fond
    return np.where(garder[etiq], 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Registre et point d'entrée
# ---------------------------------------------------------------------------

BINARISATION = {
    "fixe": seuil_fixe,
    "otsu": otsu,
    "triangle": triangle,
    "adapt_moyenne": adaptative_moyenne,
    "adapt_gauss": adaptative_gaussienne,
    "niblack": niblack,
    "sauvola": sauvola,
}

GLOBALES = ("fixe", "otsu", "triangle")


def binariser(gris, methode="sauvola", polarite="auto", nettoyage=True, **params):
    """Point d'entrée de l'étape 3.

    1. met le texte en foncé (polarité) ;
    2. seuille avec la méthode choisie (params transmis, ex. fenetre=41, k=0.3) ;
    3. nettoie les petites composantes parasites.
    Renvoie une image binaire, texte noir sur fond blanc."""
    if methode not in BINARISATION:
        raise KeyError(f"Binarisation inconnue : {methode} ({list(BINARISATION)})")
    gris, _ = normaliser_polarite(gris, polarite)
    b = BINARISATION[methode](gris, **params)
    return nettoyer(b) if nettoyage else b

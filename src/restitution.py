# -*- coding: utf-8 -*-
"""
restitution.py — Étape 6 du pipeline : restituer le texte lu à l'utilisateur.

Le cahier des charges demande une restitution sous forme de texte affiché.
Pour un public malvoyant, l'affichage suit les recommandations usuelles
d'accessibilité (WCAG 2.x, critères 1.4.3 et 1.4.6 ; recommandations de
lisibilité basse vision) :
  - très grand corps de caractères (≥ 48 px ici) ;
  - contraste maximal : texte jaune ou blanc sur fond noir (rapport > 15:1),
    configuration souvent préférée en basse vision car elle limite
    l'éblouissement ;
  - police sans empattement, une information par ligne, lignes courtes ;
  - pas d'information portée uniquement par la couleur.

Trois sorties :
  1. texte brut nettoyé (pour une lecture d'écran / synthèse vocale) ;
  2. affichage console en grandes lignes séparées ;
  3. « carte de lecture » PNG : le texte en très grand, fort contraste, avec
     en vignette la photo et les zones lues numérotées (pour qu'un
     accompagnant puisse vérifier d'où vient chaque ligne).

Le post-traitement nettoie la sortie de Tesseract :
  - suppression des mots isolés d'un seul caractère non alphanumérique ;
  - suppression des lignes identiques répétées (même panneau détecté deux
    fois à deux échelles) ;
  - réduction des espaces.
"""

import re
import textwrap

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOND = "#000000"
TEXTE = "#FFE600"          # jaune vif sur noir : contraste ≈ 17:1
SECONDAIRE = "#FFFFFF"


def nettoyer_ligne(ligne):
    mots = [m for m in ligne.split() if any(c.isalnum() for c in m)]
    ligne = " ".join(mots)
    return re.sub(r"\s+", " ", ligne).strip()


def _ligne_utile(ligne, longueur_min=3):
    """Une ligne est gardée si elle contient un mot d'au moins longueur_min
    caractères ou un chiffre (numéros de bâtiment : « 7 », routes : « D2209 »).
    Les lignes faites uniquement de fragments de 1-2 lettres (« SS », « L' »)
    sont presque toujours des lectures parasites de bords ou de pictogrammes."""
    return any(len(m) >= longueur_min for m in ligne.split()) or any(c.isdigit() for c in ligne)


def post_traiter(regions, longueur_min=3):
    """Nettoie et dédoublonne les lectures, dans l'ordre de lecture.

    regions : liste de (boîte, texte lu) telle que renvoyée par l'OCR.
    Renvoie la liste des (ligne, boîte) conservées : chaque ligne restituée
    reste associée à la zone de l'image d'où elle provient."""
    vues, sortie = set(), []
    for boite, texte in regions:
        ligne = nettoyer_ligne(texte or "")
        cle = ligne.lower()
        if ligne and _ligne_utile(ligne, longueur_min) and cle not in vues:
            vues.add(cle)
            sortie.append((ligne, boite))
    return sortie


def afficher_console(lignes, titre="TEXTE LU"):
    """Affichage terminal : une information par ligne, bien séparée."""
    barre = "═" * 60
    print(f"\n{barre}\n  {titre}\n{barre}")
    if not lignes:
        print("\n  (aucun texte lisible détecté)\n")
    for i, ligne in enumerate(lignes, 1):
        print(f"\n  {i}.  {ligne.upper()}")
    print(f"\n{barre}\n")


def carte_de_lecture(couleur, lectures, chemin, max_lignes=8):
    """Image PNG fort contraste : texte en très grand à gauche, photo avec les
    zones numérotées à droite. Le numéro d'une ligne est celui de sa zone.

    lectures : liste de (ligne, boîte) renvoyée par post_traiter."""
    affichees = lectures[:max_lignes]
    fig = plt.figure(figsize=(16, 9), facecolor=FOND)
    ax_t = fig.add_axes([0.02, 0.05, 0.56, 0.9], facecolor=FOND)
    ax_t.axis("off")
    if not affichees:
        ax_t.text(0, 0.5, "Aucun texte\nlisible détecté", color=TEXTE, fontsize=54,
                  fontweight="bold", va="center", family="DejaVu Sans")
    else:
        taille = 54 if len(affichees) <= 4 else (44 if len(affichees) <= 6 else 36)
        pas = 1.0 / (len(affichees) + (1 if len(lectures) > max_lignes else 0))
        for i, (ligne, _) in enumerate(affichees):
            texte = "\n".join(textwrap.wrap(ligne.upper(), 18)) or ligne
            ax_t.text(0, 1 - (i + 0.5) * pas, f"{i + 1}  {texte}", color=TEXTE, fontsize=taille,
                      fontweight="bold", va="center", family="DejaVu Sans")
        if len(lectures) > max_lignes:
            ax_t.text(0, 0.5 * pas, f"+ {len(lectures) - max_lignes} autre(s) ligne(s) (voir le .txt)",
                      color=SECONDAIRE, fontsize=22, va="center", family="DejaVu Sans")

    ax_i = fig.add_axes([0.6, 0.1, 0.38, 0.8])
    img = cv2.cvtColor(couleur, cv2.COLOR_BGR2RGB).copy()
    ep = max(3, img.shape[1] // 250)
    for i, (_, (x, y, w, h)) in enumerate(affichees, 1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 230, 0), ep)
        cv2.putText(img, str(i), (x, max(ep * 6, y - ep * 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    ep / 2.5, (255, 230, 0), ep, cv2.LINE_AA)
    ax_i.imshow(img)
    ax_i.axis("off")
    ax_i.set_title("Zones lues", color=SECONDAIRE, fontsize=20)
    fig.savefig(chemin, dpi=80, facecolor=FOND)
    plt.close(fig)

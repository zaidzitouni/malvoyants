#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telecharger_commons.py
----------------------
Constitue rapidement une base d'images de texte urbain (plaques de rue,
numéros de bâtiments, panneaux) à partir de Wikimedia Commons, en ne
gardant QUE des images sous licence ouverte réutilisable
(CC0, domaine public, CC BY, CC BY-SA — on exclut NC et ND).

Pour chaque image téléchargée, une ligne est ajoutée à SOURCES.csv
(fichier, titre, auteur, licence, URL de la page) : c'est la traçabilité
exigée par les licences CC BY / CC BY-SA.

Installation :
    pip install requests

Exemples :
    # Toute la base d'un coup (requêtes prédéfinies ci-dessous)
    python telecharger_commons.py --preset --max 15

    # Une catégorie précise
    python telecharger_commons.py --categorie "House numbers in France" --max 20 --sous-dossier numeros

    # Une recherche par mots-clés
    python telecharger_commons.py --recherche "plaque de rue" --max 20 --sous-dossier plaques
"""

import argparse
import csv
import html
import re
import sys
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia exige un User-Agent identifiant l'outil (politique d'usage de l'API).
HEADERS = {
    "User-Agent": "ProjetOCR-Malvoyants/1.0 (projet academique; python-requests)"
}

# Requêtes prédéfinies : (sous-dossier, type, valeur)
# type = "categorie" ou "recherche". Les recherches fonctionnent même si une
# catégorie a été renommée sur Commons.
PRESET = [
    ("numeros",  "categorie", "House numbers in France"),
    ("numeros",  "recherche", "house number plate door"),
    ("plaques",  "recherche", "plaque de rue"),
    ("plaques",  "recherche", "street name sign Paris"),
    ("plaques",  "recherche", "street sign Morocco"),
    ("panneaux", "recherche", "panneau signalisation direction ville"),
    ("panneaux", "recherche", "information sign street"),
    ("affiches", "recherche", "shop sign storefront text"),
]

# Taille de la vignette téléchargée (px de large). 1600 px suffit pour l'OCR
# et évite de télécharger des originaux de 15 Mo.
LARGEUR = 1600


def licence_acceptee(nom_licence: str) -> bool:
    """Garde CC0, domaine public, CC BY, CC BY-SA ; rejette NC et ND."""
    n = (nom_licence or "").upper()
    jetons = re.split(r"[\s\-]+", n)
    if "NC" in jetons or "ND" in jetons:
        return False
    return n.startswith("CC BY") or n.startswith("CC-BY") or "CC0" in n \
        or "PUBLIC DOMAIN" in n or n.startswith("PD")


def nettoyer_html(texte: str) -> str:
    """Le champ Artist de Commons contient du HTML : on garde le texte brut."""
    return html.unescape(re.sub(r"<[^>]+>", "", texte or "")).strip()


def interroger(params_generateur: dict, maximum: int):
    """Itère sur les fichiers renvoyés par l'API (avec gestion de la pagination)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": LARGEUR,
        **params_generateur,
    }
    obtenus = 0
    while obtenus < maximum:
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        donnees = r.json()
        for page in donnees.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            yield page["title"], info
            obtenus += 1
            if obtenus >= maximum * 3:  # marge : beaucoup seront filtrés
                return
        if "continue" not in donnees:
            return
        params.update(donnees["continue"])
        time.sleep(0.5)  # courtoisie envers l'API


def telecharger(sortie: Path, sous_dossier: str, type_req: str, valeur: str,
                maximum: int, journal: csv.writer, deja_vus: set):
    if type_req == "categorie":
        gen = {"generator": "categorymembers",
               "gcmtitle": f"Category:{valeur}",
               "gcmtype": "file", "gcmlimit": 50}
    else:
        gen = {"generator": "search", "gsrsearch": f"filetype:bitmap {valeur}",
               "gsrnamespace": 6, "gsrlimit": 50}

    dossier = sortie / sous_dossier
    dossier.mkdir(parents=True, exist_ok=True)
    n = 0
    print(f"\n[{sous_dossier}] {type_req} : {valeur}")

    for titre, info in interroger(gen, maximum):
        if n >= maximum:
            break
        if titre in deja_vus:
            continue
        meta = info.get("extmetadata", {})
        licence = meta.get("LicenseShortName", {}).get("value", "")
        if not licence_acceptee(licence):
            continue

        url_img = info.get("thumburl") or info.get("url")
        ext = ".png" if info.get("mime") == "image/png" else ".jpg"
        nom = f"{sous_dossier}_{len(list(dossier.glob('*'))) + 1:03d}{ext}"

        try:
            img = requests.get(url_img, headers=HEADERS, timeout=60)
            img.raise_for_status()
        except requests.RequestException as e:
            print(f"   échec {titre} : {e}")
            continue

        (dossier / nom).write_bytes(img.content)
        deja_vus.add(titre)
        journal.writerow([
            f"{sous_dossier}/{nom}", titre,
            nettoyer_html(meta.get("Artist", {}).get("value", "")),
            licence,
            meta.get("LicenseUrl", {}).get("value", ""),
            info.get("descriptionurl", ""),
        ])
        n += 1
        print(f"   ✓ {nom}  ({licence})")
        time.sleep(0.3)

    print(f"   → {n} image(s) retenue(s)")


def main():
    p = argparse.ArgumentParser(description="Téléchargement d'images libres depuis Wikimedia Commons")
    p.add_argument("--preset", action="store_true", help="exécuter toutes les requêtes prédéfinies")
    p.add_argument("--categorie", help="nom de catégorie Commons (sans 'Category:')")
    p.add_argument("--recherche", help="mots-clés de recherche")
    p.add_argument("--sous-dossier", default="divers")
    p.add_argument("--max", type=int, default=15, help="images max par requête")
    p.add_argument("--sortie", default="data/images")
    args = p.parse_args()

    if not (args.preset or args.categorie or args.recherche):
        p.error("indiquez --preset, --categorie ou --recherche")

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    chemin_csv = sortie.parent / "SOURCES.csv"
    nouveau = not chemin_csv.exists()

    with open(chemin_csv, "a", newline="", encoding="utf-8") as f:
        journal = csv.writer(f)
        if nouveau:
            journal.writerow(["fichier", "titre_commons", "auteur",
                              "licence", "url_licence", "page_source"])
        deja_vus: set = set()
        if args.preset:
            for sous, type_req, valeur in PRESET:
                telecharger(sortie, sous, type_req, valeur, args.max, journal, deja_vus)
        if args.categorie:
            telecharger(sortie, args.sous_dossier, "categorie", args.categorie,
                        args.max, journal, deja_vus)
        if args.recherche:
            telecharger(sortie, args.sous_dossier, "recherche", args.recherche,
                        args.max, journal, deja_vus)

    print(f"\nTerminé. Traçabilité des licences : {chemin_csv}")


if __name__ == "__main__":
    sys.exit(main())
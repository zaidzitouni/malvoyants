# -*- coding: utf-8 -*-
"""
evaluation.py — Métriques de reconnaissance de texte.

Trois métriques complémentaires, toutes calculées après normalisation
(minuscules, espaces et retours à la ligne réduits à un espace ; accents
conservés : « Percée » lu « Percee » est une vraie erreur) :

CER  (Character Error Rate) = distance de Levenshtein au niveau des
     caractères / nombre de caractères de la référence. 0 = parfait.
     Peut dépasser 1 : chaque caractère inventé (insertion) compte, et
     Tesseract en invente beaucoup dans les arbres, façades, etc.

WER  (Word Error Rate) : même calcul au niveau des mots. Plus sévère :
     une seule lettre fausse rend tout le mot faux.

F1 MOTS (indépendant de l'ordre) : on compare les MULTI-ENSEMBLES de mots
     lus et attendus. précision = mots lus corrects / mots lus ;
     rappel = mots attendus retrouvés / mots attendus ; F1 = moyenne
     harmonique. Utile pour les scènes à plusieurs panneaux : Tesseract ne
     lit pas forcément les panneaux dans l'ordre de la vérité terrain, et
     CER/WER pénalisent alors un texte pourtant correctement lu.
     Le RAPPEL répond directement à la question de l'utilisateur
     malvoyant : « quelle part de l'information affichée m'est restituée ? »
"""

from collections import Counter


def normaliser(texte):
    return " ".join((texte or "").lower().split())


def levenshtein(a, b):
    """Distance d'édition (insertions, suppressions, substitutions).
    Fonctionne sur des chaînes (CER) ou des listes de mots (WER)."""
    if len(a) < len(b):
        a, b = b, a
    prec = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cour = [i]
        for j, cb in enumerate(b, 1):
            cour.append(min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (ca != cb)))
        prec = cour
    return prec[-1]


def cer(reference, hypothese):
    r, h = normaliser(reference), normaliser(hypothese)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def wer(reference, hypothese):
    r, h = normaliser(reference).split(), normaliser(hypothese).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def scores_mots(reference, hypothese):
    """(précision, rappel, F1) sur les mots, sans tenir compte de l'ordre."""
    r = Counter(normaliser(reference).split())
    h = Counter(normaliser(hypothese).split())
    communs = sum((r & h).values())
    precision = communs / sum(h.values()) if h else (1.0 if not r else 0.0)
    rappel = communs / sum(r.values()) if r else 1.0
    f1 = 0.0 if precision + rappel == 0 else 2 * precision * rappel / (precision + rappel)
    return precision, rappel, f1


def evaluer(reference, hypothese):
    """Toutes les métriques d'un coup (dictionnaire prêt pour un CSV)."""
    p, r, f = scores_mots(reference, hypothese)
    return {"cer": cer(reference, hypothese), "wer": wer(reference, hypothese),
            "precision_mots": p, "rappel_mots": r, "f1_mots": f}

"""Construction du dataset : une ligne = un séisme, colonnes =

Colonnes :
  - META_COLS : identifiants et paramètres VRAIS de la source -> jamais donnés au modèle
                (servent au regroupement pour la validation croisée et à l'analyse des erreurs)
  - TARGET    : "mw" (magnitude finale)
  - le reste  : caractéristiques calculées par src/features.py (entrées du modèle)
"""
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import generate as G
from . import pegs_io as io

TARGET = "mw"
GROUP = "noise_window"   # groupe pour la validation croisée (voir notebook 02, section 4)
META_COLS = ["event_id", "split", "noise_window", "noise_start", "mecanisme", "src_id",
             "srclat", "srclon", "strike", "dip", "rake"]


def build_synthetic(split, kernels=None, noise=None):
    """Génère les séismes d'un split ('train' ou 'test') et calcule leurs caractéristiques."""
    kernels = kernels if kernels is not None else io.load_kernels(C.KERNEL_FILE)
    noise = noise if noise is not None else io.load_noise(C.NOISE_FILE)
    n, windows, seed = ((C.N_TRAIN, C.NOISE_TRAIN, C.SEED_TRAIN) if split == "train"
                        else (C.N_TEST, C.NOISE_TEST, C.SEED_TEST))
    rows = []
    for ev in G.iter_events(n, windows, kernels, noise, seed=seed, mw_range=C.MW_RANGE):
        row = F.event_features(ev["data"], ev["tp"], ev["flag"], T=C.T_DECISION)
        row.update({k: ev[k] for k in ["noise_window", "noise_start", "mecanisme", "src_id",
                                        "srclat", "srclon", "strike", "dip", "rake", "mw"]})
        row["event_id"] = f"{split}_{ev['event_id']:05d}"
        row["split"] = split
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[META_COLS + [TARGET] + [c for c in df.columns if c not in META_COLS + [TARGET]]]


def build_real():
    """Caractéristiques des 4 séismes réels (test final hors distribution)."""
    ref = io.load_stations(C.KERNEL_FILE).station.values
    rows = []
    for p in sorted(C.REAL_DIR.glob("*.h5")):
        ev = io.load_real_event(p, ref)
        row = F.event_features(ev["data"], ev["tp"], ev["flag"], T=C.T_DECISION)
        row.update({"event_id": ev["event"], "split": "real", "mw": ev["mw"], "srclat": ev["srclat"],
                    "srclon": ev["srclon"], "strike": ev["strike"], "dip": ev["dip"], "rake": ev["rake"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in META_COLS:
        if c not in df:
            df[c] = np.nan
    return df[META_COLS + [TARGET] + [c for c in df.columns if c not in META_COLS + [TARGET]]]


def feature_columns(df, spatial=True):
    """Liste des colonnes d'entrée du modèle (famille A, et B si spatial=True)."""
    cols = [c for c in df.columns if c not in META_COLS + [TARGET]]
    return cols if spatial else [c for c in cols if not F.is_spatial(c)]

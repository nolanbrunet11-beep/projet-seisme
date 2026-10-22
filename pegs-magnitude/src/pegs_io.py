"""Lecture des fichiers HDF5 (.h5) de PEGSGraph.

Chaque fonction renvoie des tableaux NumPy simples (et un DataFrame pandas pour les métadonnées),
pour que le reste du projet n'ait pas à connaître la structure interne des fichiers.

Unités (vérifiées dans le notebook 01) :
- signaux en accélération verticale, m/s² (1 nm/s² = 1e-9 m/s²) ;
- temps en secondes, échantillonnage 1 Hz ;
- tp = temps d'arrivée de l'onde P à chaque station (s après l'origine).
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _decode(arr):
    """Convertit un tableau de bytes HDF5 (b'AK.ANM') en chaînes Python."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def describe_h5(path):
    """Liste tous les jeux de données d'un fichier .h5 : nom, forme, type."""
    rows = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append({"fichier": Path(path).name, "dataset": name,
                             "forme": obj.shape, "type": str(obj.dtype)})
        f.visititems(visit)
    return pd.DataFrame(rows)


def load_stations(path):
    """Codes et coordonnées des stations (présents dans tous les fichiers)."""
    with h5py.File(path, "r") as f:
        return pd.DataFrame({"station": _decode(f["stacode"][:]),
                             "lat": f["stalat"][:], "lon": f["stalon"][:]})


def load_pegs_db(path):
    """Base de PEGS SANS bruit (fichier 03_..._tiny.h5).

    Renvoie :
      pegs : (n_séismes, n_stations, 512) — origine à l'échantillon 0, mis à zéro après tp
      tp   : (n_séismes, n_stations)
      meta : DataFrame (mw, m0, position, mécanisme)
      momrate : (n_séismes, 351) taux de moment (fonction source), N·m/s
      mt   : (n_séismes, 6) tenseur des moments normalisé
    """
    with h5py.File(path, "r") as f:
        pegs = f["PEGS"][:, :, :, 0]
        tp = f["tp"][:]
        meta = pd.DataFrame({k: f[k][:] for k in
                             ["mw", "m0", "srclat", "srclon", "srcdep", "strike", "dip", "rake"]})
        momrate = f["momrate"][:]
        mt = f["mt"][:]
    return pegs, tp, meta, momrate, mt


def load_graph_split(path, split):
    """Un split ('train', 'valid', 'test') d'un fichier graph_db ou real_time.

    Renvoie un dict :
      data  : (n, n_stations, 601) signal BRUITÉ (ce qu'un sismomètre enregistre), origine à l'indice 300
      pegs  : idem SANS bruit (absent des fichiers real_time : on ne connaît pas le "vrai" PEGS)
      flag  : (n, n_stations) True si la station a des données
      tp, mw, srclat, srclon, srcdep, strike, dip, rake, stf, noiselevel (si présent)
    """
    out = {}
    with h5py.File(path, "r") as f:
        g = f[split]
        for k in g.keys():
            arr = g[k][:]
            if arr.ndim >= 3 and arr.shape[-1] == 1:   # retire l'axe "composante" (verticale seule)
                arr = arr[..., 0]
            out[k.lower() if k != "PEGS" else "pegs"] = arr
    out["flag"] = out["flag"].astype(bool)
    return out


def load_real_event(path, ref_codes):
    """Un séisme réel (dossier real_time/), ré-aligné sur la liste de stations de référence (273 stations).

    Le fichier de Chignik 2021 utilise la configuration « legacy » (178 stations) : les stations absentes
    sont ajoutées avec flag=False et un signal nul, pour que chaque colonne corresponde toujours à la même station.
    """
    d = load_graph_split(path, "test")
    with h5py.File(path, "r") as f:
        codes = _decode(f["stacode"][:])
    idx = {c: i for i, c in enumerate(ref_codes)}
    n_ref, n_t = len(ref_codes), d["data"].shape[2]
    data = np.zeros((n_ref, n_t), np.float32); tp = np.zeros(n_ref); flag = np.zeros(n_ref, bool)
    for j, c in enumerate(codes):
        if c in idx:
            i = idx[c]
            data[i], tp[i], flag[i] = d["data"][0, j], d["tp"][0, j], d["flag"][0, j]
    return {"data": data, "tp": tp, "flag": flag, "mw": float(d["mw"][0]),
            "srclat": float(d["srclat"][0]), "srclon": float(d["srclon"][0]),
            "strike": float(d["strike"][0]), "dip": float(d["dip"][0]), "rake": float(d["rake"][0]),
            "event": Path(path).stem}


def load_noise(path):
    """Bruit sismique réel enregistré aux stations (fichier 04_...).

    noise : (n_fenêtres, n_stations, 3600) — fenêtres d'1 h à 1 Hz, déjà filtrées 2–30 mHz
    flag  : (n_fenêtres, n_stations) True si la station a des données
    noiselevel : écart-type du bruit par trace (m/s²)
    """
    with h5py.File(path, "r") as f:
        return {"noise": f["noise"][:, :, :, 0],
                "flag": f["flag"][:, :, 0].astype(bool),
                "noiselevel": f["noiselevel"][:, :, 0],
                "nnz": f["nnz"][:, :, 0]}


def load_kernels(path):
    """Noyaux de Green PEGS (fichier 01_...) : réponse de chaque station à 6 sources élémentaires.

    ker : (n_sources, n_stations, 512, 6) ; tp : (n_sources, n_stations) ;
    srclat/srclon : position des n_sources points de la grille.
    """
    with h5py.File(path, "r") as f:
        return {"ker": f["kerPEGS"][:, :, :, 0, :], "tp": f["kertp"][:],
                "srclat": f["srclat"][:], "srclon": f["srclon"][:], "srcdep": f["srcdep"][:]}

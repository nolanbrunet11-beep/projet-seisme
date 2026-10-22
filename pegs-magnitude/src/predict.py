"""Prédiction de la magnitude à partir des signaux PEGS bruts.

Usage minimal :
    from src.predict import predict_magnitude
    mw = predict_magnitude(signal_pegs)          # signal_pegs : tableau (273 stations, 601 s)

Format attendu de `signal_pegs` (identique aux fichiers PEGSGraph `graph_db` et `real_time`) :
  - accélération verticale en m/s², échantillonnée à 1 Hz, déjà filtrée 2–30 mHz ;
  - 601 échantillons : 300 s avant l'origine du séisme, l'origine à l'indice 300, puis 300 s après ;
  - une ligne par station, dans l'ordre des 273 stations de référence (ou fournir `station_codes`) ;
  - signal mis à zéro après l'arrivée de l'onde P à chaque station ; station sans données = ligne de zéros.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import pegs_io as io

MODEL_DIR = C.PROJECT_ROOT / "models"
_CACHE = {}


def _load(model_name):
    if model_name not in _CACHE:
        path = MODEL_DIR / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable : exécuter d'abord notebooks/03_modelisation.ipynb")
        _CACHE[model_name] = joblib.load(path)
    return _CACHE[model_name]


def infer_tp_flag(signal):
    """Déduit, si on ne les fournit pas, l'arrivée P et les stations actives à partir du signal lui-même.

    - flag : la station a au moins un échantillon non nul ;
    - tp   : dernier échantillon non nul après l'origine (le signal est mis à zéro après l'onde P).
    """
    signal = np.asarray(signal)
    flag = np.abs(signal).max(1) > 0
    post = signal[:, C.GRAPH_ORIGIN_IDX:]
    nz = np.abs(post) > 0
    last = np.where(nz.any(1), post.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1), 0)
    return last.astype(float), flag


def _align(signal, tp, flag, station_codes):
    """Ré-ordonne les stations sur la liste de référence (les absentes : flag=False)."""
    ref = io.load_stations(C.KERNEL_FILE).station.values
    idx = {c: i for i, c in enumerate(ref)}
    out_s = np.zeros((len(ref), signal.shape[1]), np.float32)
    out_tp, out_f = np.zeros(len(ref)), np.zeros(len(ref), bool)
    for j, c in enumerate(station_codes):
        if c in idx:
            out_s[idx[c]], out_tp[idx[c]], out_f[idx[c]] = signal[j], tp[j], flag[j]
    return out_s, out_tp, out_f


def detection_threshold(model="xgboost"):
    """Seuil de détection (Mw) : 99e centile des prédictions du modèle sur du BRUIT PUR (calculé au notebook 04).

    Une prédiction sous ce seuil ne se distingue pas de ce que le modèle répond en l'absence de séisme visible :
    il faut alors la lire comme « Mw sous le seuil de détection des PEGS », et non comme une valeur fiable.
    """
    import json
    path = MODEL_DIR / "noise_baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(model, {}).get("p99")


def predict_magnitude(signal_pegs, tp=None, flag=None, station_codes=None, model="xgboost", return_details=False):
    """Estime la magnitude de moment Mw d'un séisme à partir de ses signaux PEGS.

    Paramètres
    ----------
    signal_pegs : array (n_stations, 601), m/s² — enregistrement du réseau (voir format en tête de fichier)
    tp : array (n_stations,), optionnel — arrivée de l'onde P (s après l'origine) ; déduite du signal si absente
    flag : array (n_stations,) booléen, optionnel — station active ; déduite du signal si absente
    station_codes : liste, optionnel — codes des stations si l'ordre diffère des 273 stations de référence
    model : "xgboost" (modèle final), "ridge" ou "dummy"
    return_details : si True, renvoie un dict avec la Mw, le seuil de détection et un indicateur de détection

    Retour
    ------
    float : magnitude Mw estimée, 300 s après l'origine (instant de décision du projet)
    (ou dict si return_details=True)
    """
    signal = np.asarray(signal_pegs, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 601:
        raise ValueError(f"signal_pegs doit avoir la forme (n_stations, 601), reçu {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("signal_pegs contient des NaN ou des infinis : remplacer les stations sans données par des zéros")
    if tp is None or flag is None:
        tp_i, flag_i = infer_tp_flag(signal)
        tp = tp_i if tp is None else np.asarray(tp, float)
        flag = flag_i if flag is None else np.asarray(flag, bool)
    if station_codes is not None:
        signal, tp, flag = _align(signal, np.asarray(tp, float), np.asarray(flag, bool), list(station_codes))
    if signal.shape[0] != 273:
        raise ValueError("273 stations attendues (ou fournir station_codes pour ré-aligner)")

    bundle = _load(model)
    feats = F.event_features(signal, tp, flag, T=bundle["T_decision"], spatial=True)
    X = pd.DataFrame([feats]).reindex(columns=bundle["features"])
    mw = float(bundle["model"].predict(X)[0])
    if not return_details:
        return mw
    thr = detection_threshold(model)
    return {"mw": mw, "seuil_detection": thr,
            "detecte": None if thr is None else bool(mw > thr),
            "n_stations_utilisees": int(feats["n_stations_ok"])}

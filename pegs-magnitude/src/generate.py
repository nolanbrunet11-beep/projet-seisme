"""Génération de séismes synthétiques PEGS.

Pourquoi générer ? Les bases prêtes à l'emploi de PEGSGraph ne contiennent que 15 séismes (« tiny »).
On reproduit la méthode des auteurs (Licciardi et al., 2022 ; Hourcade et al., 2025) avec les briques fournies :

    PEGS(station, t) = 1e-22 × [ Σ_i  mt_i × noyau_i(station, t) ]  ⊛  taux_de_moment(t)

  - noyau_i      : réponse PEGS de chaque station à 6 sources élémentaires (fichier 01_kernel, 15 positions de source)
  - mt           : tenseur des moments (type de faille), convention GCMT (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
  - taux_de_moment : fonction source temporelle (STF), modèle de Meier et al. (2017)
  - ⊛            : convolution
Cette formule a été vérifiée : elle redonne EXACTEMENT les 15 PEGS du fichier 03_PEGS_db (notebook 02).

On ajoute ensuite du BRUIT RÉEL enregistré aux stations (fichier 04_noise), puis on met le signal à zéro après
l'arrivée de l'onde P, comme dans les données réelles.
"""
import numpy as np
from scipy.signal import fftconvolve

from . import config as C

N_STF = 351          # durée de la fonction source (s), comme dans la base des auteurs
N_WIN = 601          # longueur des traces (s) : 300 s avant + 300 s après l'origine
ORIGIN = C.GRAPH_ORIGIN_IDX
KERNEL_SCALE = 1e-22  # facteur d'échelle des noyaux (retrouvé en reproduisant le fichier 03_PEGS_db)
MECHANISMS = ("chevauchement", "normal", "decrochement")


# --------------------------------------------------------------------------- source
def moment_tensor(strike, dip, rake):
    """Tenseur des moments d'un double couple (norme unité, Aki & Richards) en convention GCMT.

    Renvoie [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp] — même ordre que le dataset `mt` des fichiers PEGSGraph (vérifié).
    """
    s, d, r = np.deg2rad([strike, dip, rake])
    mxx = -(np.sin(d) * np.cos(r) * np.sin(2 * s) + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    mxy = np.sin(d) * np.cos(r) * np.cos(2 * s) + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s)
    mxz = -(np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
    myy = np.sin(d) * np.cos(r) * np.sin(2 * s) - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2
    myz = -(np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2 * d) * np.sin(r) * np.cos(s))
    mzz = np.sin(2 * d) * np.sin(r)
    # x = nord, y = est, z = bas  ->  r = haut, t = sud, p = est
    return np.array([mzz, mxx, myy, mxz, -myz, -mxy], dtype=np.float64)


def sample_mechanism(rng, kind):
    """Tire (strike, dip, rake) en degrés pour un type de faille.

    Choix simplificateurs (à citer comme hypothèse) : azimut uniforme ; pendage typique de chaque type ;
    rake gaussien (σ = 10°) autour de la valeur pure, comme Licciardi et al. (2022) pour les chevauchements.
    """
    strike = rng.uniform(0, 360)
    if kind == "chevauchement":
        dip, rake = rng.uniform(10, 40), rng.normal(90, 10)
    elif kind == "normal":
        dip, rake = rng.uniform(40, 70), rng.normal(-90, 10)
    elif kind == "decrochement":
        dip, rake = rng.uniform(60, 90), rng.choice([0.0, 180.0]) + rng.normal(0, 10)
    else:
        raise ValueError(kind)
    return strike, dip, rake


def source_time_function(rng, mw, n=N_STF):
    """Taux de moment (N·m/s, 1 Hz) selon Meier et al. (2017), équations (1)-(2) de Licciardi et al. (2022).

    f(t) = t · exp(-0.5 (λt)²) · [1 + N(t)],  λ = 10^(7.24 − 0.41 log10(M0) + ε),  ε ~ N(0, 0.15)
    N(t) = 0.38 · n(t)/σ  avec n(t) intégrale d'un bruit blanc gaussien (irrégularités des vraies ruptures).
    Normalisé pour que ∫ STF = M0, donc la magnitude finale vaut exactement `mw`.
    """
    m0 = 10 ** (1.5 * mw + 9.1)
    lam = 10 ** (7.24 - 0.41 * np.log10(m0) + rng.normal(0, 0.15))
    t = np.arange(n, dtype=np.float64)
    walk = np.cumsum(rng.normal(size=n))
    rough = 0.38 * walk / walk.std()
    f = np.clip(t * np.exp(-0.5 * (lam * t) ** 2) * (1 + rough), 0, None)
    return m0 * f / f.sum()


def synth_pegs(kernels, k, mt, momrate):
    """PEGS sans bruit (n_stations, 512), origine à l'indice 0, mis à zéro après l'onde P."""
    green = kernels["ker"][k] @ mt                                   # (n_stations, 512)
    pegs = KERNEL_SCALE * fftconvolve(green, momrate[None, :], axes=1)[:, :green.shape[1]]
    t = np.arange(green.shape[1])
    return np.where(t[None, :] <= kernels["tp"][k][:, None], pegs, 0.0)


# --------------------------------------------------------------------------- évènement complet
def make_event(rng, kernels, noise, noise_windows, mw, kind):
    """Un séisme complet tel qu'un réseau de sismomètres l'enregistrerait.

    Renvoie un dict : data (n_stations, 601) en m/s², tp, flag, et les paramètres vrais (pour l'analyse).
    """
    k = rng.integers(len(kernels["srclat"]))                        # position de la source (15 possibles)
    strike, dip, rake = sample_mechanism(rng, kind)
    mt = moment_tensor(strike, dip, rake)
    momrate = source_time_function(rng, mw)
    pegs = synth_pegs(kernels, k, mt, momrate)

    w = rng.choice(noise_windows)                                    # fenêtre de bruit réel (1 h)
    start = rng.integers(0, noise["noise"].shape[2] - N_WIN)         # même instant pour toutes les stations
    data = noise["noise"][w, :, start:start + N_WIN].astype(np.float64)   # (cohérence spatiale du bruit)
    data[:, ORIGIN:] += pegs[:, :N_WIN - ORIGIN]

    tp = kernels["tp"][k].astype(np.float64)
    t_rel = np.arange(N_WIN) - ORIGIN
    data = np.where(t_rel[None, :] > tp[:, None], 0.0, data)         # zéro après l'arrivée P
    flag = noise["flag"][w].copy()
    data[~flag] = 0.0                                                # station sans données
    return {"data": data.astype(np.float32), "tp": tp, "flag": flag,
            "mw": mw, "mecanisme": kind, "strike": strike, "dip": dip, "rake": rake,
            "src_id": int(k), "srclat": float(kernels["srclat"][k]), "srclon": float(kernels["srclon"][k]),
            "noise_window": int(w), "noise_start": int(start), "momrate": momrate}


def iter_events(n_events, noise_windows, kernels, noise, seed, mw_range=(7.0, 9.5)):
    """Générateur de `n_events` séismes : Mw uniforme, types de faille en proportions égales."""
    rng = np.random.default_rng(seed)
    kinds = np.resize(MECHANISMS, n_events)
    rng.shuffle(kinds)
    for i in range(n_events):
        ev = make_event(rng, kernels, noise, noise_windows, rng.uniform(*mw_range), kinds[i])
        ev["event_id"] = i
        yield ev

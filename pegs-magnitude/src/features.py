"""Prétraitement des signaux PEGS et extraction de caractéristiques.

Entrée : l'enregistrement d'UN séisme sur tout le réseau
    data (n_stations, 601) en m/s², origine à l'indice 300 ; tp (n_stations,) en s ; flag (n_stations,) booléen
Sortie : UNE ligne (dict) de caractéristiques numériques -> exploitable par XGBoost / Ridge.

Uniquement des informations disponibles au moment de la décision (T secondes après l'origine) :
le signal bruité avant l'onde P, le bruit AVANT le séisme, et les temps d'arrivée P (détecteur d'onde P).

Deux familles de caractéristiques :
  A. Agrégats sur le réseau (≈ 50 colonnes) : statistiques robustes (médiane, quantile 90 %, max…) de grandeurs
     calculées station par station — amplitude, rapport signal/bruit, pente, énergie par bande de fréquence.
  B. Motif spatial (1 colonne par station) : valeur moyenne SIGNÉE du signal juste avant l'onde P, divisée par
     le bruit de la station. Le signe et la répartition géographique dépendent du type de faille ; on laisse le
     modèle s'en servir. (Le choix A seul / A+B sera tranché en validation croisée, étape 8.)

Pourquoi pas de MFCC ? L'échelle de Mel imite l'oreille humaine et est quasi LINÉAIRE en dessous de ~1 000 Hz.
Nos signaux sont dans la bande 2–30 mHz (échantillonnage 1 Hz) : les MFCC s'y réduisent à un simple
cepstre linéaire, sans intérêt physique par rapport aux énergies par bande calculées ici directement.
"""
import numpy as np

from . import config as C

CLIP = 10e-9            # écrêtage à ±10 nm/s² (Licciardi et al., 2022) : coupe les pics aberrants du bruit
NM = 1e9                # conversion m/s² -> nm/s²
MIN_LEN = 10            # une station n'est exploitée que si elle a au moins 10 s de signal avant l'onde P
END_WIN = 30            # « fin de signal » = 30 dernières secondes avant l'onde P
SLOPE_WIN = 60          # pente estimée sur les 60 dernières secondes
NFFT = 1024
BANDS_MHZ = [(1, 5), (5, 10), (10, 20), (20, 40)]
EPS = 1e-12


def _band_energy(x, mask):
    """Énergie moyenne par bande de fréquence pour chaque station (x : (n, L), mask : (n, L))."""
    xm = np.where(mask, x, 0.0)
    xm = xm - (xm.sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)) * mask   # retire la moyenne
    spec = np.abs(np.fft.rfft(xm * np.hanning(x.shape[1])[None, :], n=NFFT, axis=1)) ** 2
    spec /= np.maximum(mask.sum(1, keepdims=True), 1)
    f_mhz = np.fft.rfftfreq(NFFT, d=1 / C.FS) * 1e3
    return np.stack([spec[:, (f_mhz >= lo) & (f_mhz < hi)].mean(1) for lo, hi in BANDS_MHZ], axis=1)


def station_quantities(data, tp, flag, T=300):
    """Grandeurs physiques station par station (tableau de dicts de vecteurs de taille n_stations).

    Étapes de prétraitement :
      1. écrêtage à ±10 nm/s² et conversion en nm/s² ;
      2. séparation : bruit AVANT l'origine (t < 0) / signal utile de 0 à min(tp, T) ;
      3. les stations sans données (flag=False) ou avec < 10 s de signal utile sont mises à NaN.
    """
    x = np.clip(np.asarray(data, dtype=np.float64), -CLIP, CLIP) * NM
    o = C.GRAPH_ORIGIN_IDX
    pre = x[:, :o]                                              # 300 s de bruit avant le séisme
    post = x[:, o:o + T + 1]                                    # signal après l'origine, jusqu'à T
    j = np.arange(post.shape[1])[None, :]
    t_end = np.minimum(np.floor(np.asarray(tp, float)), T)[:, None]  # dernier instant exploitable
    mask = j <= t_end
    L = mask.sum(1)
    # Règle de qualité : une station saturée (> 5 % d'échantillons écrêtés, avant OU après l'origine) est exclue,
    # sinon son bruit apparent après écrêtage serait quasi nul et le rapport signal/bruit absurde.
    clipped = np.abs(x) >= CLIP * NM
    frac_clip_pre = clipped[:, :o].mean(1)
    frac_clip_post = (clipped[:, o:o + T + 1] & mask).sum(1) / np.maximum(L, 1)
    noise_std = pre.std(1)
    ok = (np.asarray(flag, bool) & (L >= MIN_LEN) & (frac_clip_pre < 0.05) & (frac_clip_post < 0.05)
          & (noise_std > 1e-3))
    post_m = np.where(mask, post, np.nan)
    rms = np.sqrt(np.nanmean(post_m ** 2, 1))
    absmax = np.nanmax(np.abs(post_m), 1)

    end_mask = mask & (j > t_end - END_WIN)
    end_mean = np.nanmean(np.where(end_mask, post, np.nan), 1)

    sl_mask = mask & (j > t_end - SLOPE_WIN)                    # pente par moindres carrés (nm/s² par minute)
    tj = np.where(sl_mask, j, np.nan); yj = np.where(sl_mask, post, np.nan)
    tc = tj - np.nanmean(tj, 1, keepdims=True); yc = yj - np.nanmean(yj, 1, keepdims=True)
    slope = np.nansum(tc * yc, 1) / np.maximum(np.nansum(tc ** 2, 1), EPS) * 60

    e_post = _band_energy(post, mask)
    e_pre = _band_energy(pre, np.ones_like(pre, bool))

    with np.errstate(divide="ignore", invalid="ignore"):
        q = {
            "log_rms": np.log10(rms + EPS),
            "log_absmax": np.log10(absmax + EPS),
            "log_noise": np.log10(noise_std + EPS),
            "log_snr": np.log10(rms / (noise_std + EPS) + EPS),
            "end_snr": np.abs(end_mean) / (noise_std + EPS),
            "slope_snr": np.abs(slope) / (noise_std + EPS),
            "signed_end_snr": end_mean / (noise_std + EPS),
            "len_s": L.astype(float),
        }
        for b, (lo, hi) in enumerate(BANDS_MHZ):
            q[f"log_e_{lo}_{hi}mHz"] = np.log10(e_post[:, b] + EPS)
            q[f"log_bsnr_{lo}_{hi}mHz"] = np.log10((e_post[:, b] + EPS) / (e_pre[:, b] + EPS))
    for k in q:
        q[k] = np.where(ok, q[k], np.nan)
    q["ok"] = ok
    return q


AGG_QUANTITIES = ["log_rms", "log_absmax", "log_snr", "end_snr", "slope_snr"] + \
                 [f"log_e_{lo}_{hi}mHz" for lo, hi in BANDS_MHZ] + [f"log_bsnr_{lo}_{hi}mHz" for lo, hi in BANDS_MHZ]


def event_features(data, tp, flag, T=300, spatial=True):
    """Transforme l'enregistrement d'un séisme (réseau entier) en un dict de caractéristiques."""
    q = station_quantities(data, tp, flag, T)
    ok = q["ok"]
    feats = {"n_stations_ok": int(ok.sum()),
             "median_log_noise": float(np.nanmedian(q["log_noise"])) if ok.any() else np.nan,
             "median_len_s": float(np.nanmedian(q["len_s"])) if ok.any() else np.nan}
    for name in AGG_QUANTITIES:
        v = q[name][ok]
        if v.size == 0:
            for s in ("med", "p90", "max", "mean"):
                feats[f"{name}__{s}"] = np.nan
            continue
        feats[f"{name}__med"] = float(np.median(v))
        feats[f"{name}__p90"] = float(np.quantile(v, 0.9))
        feats[f"{name}__max"] = float(v.max())
        feats[f"{name}__mean"] = float(v.mean())
    with np.errstate(invalid="ignore"):
        snr = np.where(ok, 10 ** q["log_snr"], np.nan)
        for thr in (1.5, 2, 3):
            feats[f"n_snr_gt_{thr}"] = int(np.nansum(snr > thr))
            feats[f"frac_snr_gt_{thr}"] = float(np.nansum(snr > thr) / max(ok.sum(), 1))
    if spatial:
        for s, v in enumerate(q["signed_end_snr"]):
            feats[f"sta{s:03d}_endsnr"] = float(v)
    return feats


def is_spatial(col):
    """True pour les colonnes de la famille B (motif spatial, une par station)."""
    return col.startswith("sta") and col.endswith("_endsnr")

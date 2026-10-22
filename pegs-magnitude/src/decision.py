"""Décision d'alerte, coûts et ROI.

Toutes les valeurs ci-dessous sont des HYPOTHÈSES explicites (sourcées quand c'est possible) ; elles sont
regroupées ici pour pouvoir être modifiées et testées en analyse de sensibilité.

Problème de décision : un séisme de Mw ≥ 7 vient de se produire dans la zone couverte par le réseau d'Alaska.
À T = 300 s, faut-il déclencher l'ALERTE TSUNAMI MAJEURE (évacuation côtière large) ?
  - « séisme majeur » = Mw ≥ 8,5 (hypothèse : le tsunami est inexistant sous Mw 8 et très destructeur au-delà de 9
    d'après le mémoire de Q. Bletery, §I.4.2 ; 8,5 est le milieu de cette zone de transition).
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ hypothèses (scénario central)
HYP = {
    # Fréquence des séismes en Alaska (Alaska Earthquake Center, moyennes depuis 1900) :
    # « un séisme de magnitude 7 à 8 par an » et « un séisme de magnitude ≥ 8 tous les 13 ans ».
    "N_ge7_par_an": 1 + 1 / 13,                       # ≈ 1,08 séisme de Mw ≥ 7 par an
    "N_ge8_par_an": 1 / 13,
    "seuil_majeur": 8.5,
    # Coût d'une évacuation tsunami (alerte majeure) : 30 M$ (D. Walker, Honolulu Star-Bulletin, 1999 :
    # « a false warning costs the state about $30 million ») ; NOAA/ITIC : 40 M$ en 1986, 75 M$ en dollars de 2013.
    "cout_evacuation_M$": 30.0,
    # Coût d'une alerte majeure en RETARD (séisme majeur non identifié à 5 min, alerte ~20 min plus tard avec la phase W) :
    # victimes supplémentaires évitables × valeur statistique d'une vie (US DOT : 14,2 M$, dollars 2025).
    "victimes_evitables_par_retard": 10,              # HYPOTHÈSE NON SOURCÉE -> testée de 1 à 100
    "VSL_M$": 14.2,
    # Coût du projet (HYPOTHÈSES) : développement 2 data scientists × 3 mois, intégration temps réel et validation
    # par un sismologue, exploitation (calcul, maintenance, ré-entraînement annuel).
    "cout_developpement_k$": 60.0,
    "cout_integration_k$": 60.0,
    "cout_exploitation_k$_par_an": 30.0,
}


def gr_b_value(h):
    """b-value de Gutenberg-Richter déduite des deux taux observés (≥ 7 et ≥ 8)."""
    return np.log10(h["N_ge7_par_an"] / h["N_ge8_par_an"])


def annual_weights(mw, h):
    """Poids annuels des séismes de test : la base est UNIFORME en Mw, la réalité suit Gutenberg-Richter.

    Chaque séisme de test i reçoit un poids w_i ∝ 10^(-b·Mw_i), normalisé pour que Σ w_i = nombre attendu de
    séismes par an avec 7 ≤ Mw ≤ 9,5. Ainsi Σ w_i·[événement] = fréquence annuelle attendue de cet événement.
    """
    b = gr_b_value(h)
    n_7_95 = h["N_ge7_par_an"] * (1 - 10 ** (-b * 2.5))
    w = 10 ** (-b * np.asarray(mw))
    return w / w.sum() * n_7_95


def confusion(mw, pred, seuil_alerte, h, weights=None):
    """Matrice de confusion (comptes, ou fréquences annuelles si weights est donné)."""
    w = np.ones(len(mw)) if weights is None else weights
    majeur = np.asarray(mw) >= h["seuil_majeur"]
    alerte = np.asarray(pred) >= seuil_alerte
    return {"VP": w[majeur & alerte].sum(), "FN": w[majeur & ~alerte].sum(),
            "FP": w[~majeur & alerte].sum(), "VN": w[~majeur & ~alerte].sum()}


def unit_costs(h):
    c_evac = h["cout_evacuation_M$"]
    c_retard = h["victimes_evitables_par_retard"] * h["VSL_M$"]
    # VP : évacuation justifiée ; FP : évacuation inutile ; FN : évacuation quand même, mais en retard (+ victimes)
    return {"VP": c_evac, "FP": c_evac, "FN": c_evac + c_retard, "VN": 0.0}


def annual_cost(cm, h):
    c = unit_costs(h)
    return sum(cm[k] * c[k] for k in cm)


def policies(mw, preds, h, seuils=(8.0, 8.2, 8.3, 8.5)):
    """Coût annuel attendu (M$/an) de chaque politique de décision."""
    w = annual_weights(mw, h)
    majeur = np.asarray(mw) >= h["seuil_majeur"]
    n_all, n_maj = w.sum(), w[majeur].sum()
    c = unit_costs(h)
    rows = [
        {"politique": "Sans modèle A — évacuer pour tout Mw ≥ 7 (prudente)", "VP": n_maj, "FP": n_all - n_maj, "FN": 0.0, "VN": 0.0},
        {"politique": "Sans modèle B — attendre la phase W (~20 min)", "VP": 0.0, "FP": 0.0, "FN": n_maj, "VN": n_all - n_maj},
        {"politique": "Oracle (magnitude parfaite à 5 min)", "VP": n_maj, "FP": 0.0, "FN": 0.0, "VN": n_all - n_maj},
    ]
    for s in seuils:
        rows.append({"politique": f"XGBoost — alerte si Mw prédite ≥ {s}", **confusion(mw, preds, s, h, w)})
    df = pd.DataFrame(rows).set_index("politique")
    df["coût annuel attendu (M$/an)"] = sum(df[k] * c[k] for k in ["VP", "FP", "FN", "VN"])
    return df


def project_cost_year1(h):
    return (h["cout_developpement_k$"] + h["cout_integration_k$"] + h["cout_exploitation_k$_par_an"]) / 1000  # M$


def roi(gain_annuel, h):
    cp = project_cost_year1(h)
    return {"coût projet an 1 (M$)": cp, "gain annuel attendu (M$)": gain_annuel,
            "ROI an 1 (%)": 100 * (gain_annuel - cp) / cp,
            "délai de retour (mois)": cp / (gain_annuel / 12) if gain_annuel > 0 else np.inf}

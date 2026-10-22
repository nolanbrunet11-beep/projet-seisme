# Estimation de la magnitude des grands séismes à partir des PEGS

> Projet final — ECAM-EPMI.
## Problème
Les systèmes d'alerte fondés sur les ondes P **sous-estiment** la magnitude des très grands séismes (saturation vers Mw 8).
Les *Prompt Elasto-Gravity Signals* (PEGS) arrivent à la vitesse de la lumière et ne saturent pas.
Objectif : estimer **Mw** à partir des PEGS enregistrés sur le réseau de stations d'Alaska, pour éclairer la décision d'alerte tsunami.

## Données
- Source : dossier `data/` du dépôt **PEGSGraph** — <https://gitlab.com/kjuhel/pegsgraph> (GPL-3.0), consulté le 11/10/2026.
- Article : Hourcade, Juhel & Bletery (2025), *JGR: Machine Learning and Computation*, <https://doi.org/10.1029/2024JH000360>.
- Données **synthétiques** (noyaux de Green QSSP, modèle AK135) + **bruit sismique réel** + **4 séismes réels** (2018–2021).

### Installation des données
Les fichiers dépassent 50 Mo et ne sont pas versionnés. Copier le dossier `data/` de PEGSGraph à la racine du projet :
```
pegs-magnitude/
├── data/
│   ├── PEGSgraph_dataset/   (01_kernel..., 03_PEGS_db..., 04_noise_db..., graph_db..., QSSP/)
│   └── real_time/           (4 séismes réels)
```
ou définir la variable d'environnement `PEGS_DATA_DIR` vers ce dossier.

## Lancer (dans l'ordre, « Restart & Run All » sur chaque notebook)
```bash
pip install -r requirements.txt
jupyter notebook
#  1. notebooks/01_cadrage_eda.ipynb          analyse des données (~10 s)
#  2. notebooks/02_generation_features.ipynb  génère 6 000 séismes + caractéristiques -> data_processed/ (~1 min 30)
#  3. notebooks/03_modelisation.ipynb         validation croisée groupée, optimisation, modèles -> models/ (~20 min sur 2 cœurs)
#  4. notebooks/04_evaluation_erreurs.ipynb   évaluation unique sur le test, analyse des erreurs, predict_magnitude (~1 min)
#  5. notebooks/05_decision_roi_risques.ipynb  matrice de coûts, choix du seuil, ROI, sensibilité, stress tests, risques (~1 min)
```

## Utiliser le modèle
```python
from src.predict import predict_magnitude
mw = predict_magnitude(signal_pegs)                        # signal_pegs : (273 stations, 601 s), m/s², origine à l'indice 300
info = predict_magnitude(signal_pegs, return_details=True) # {'mw': ..., 'seuil_detection': 7.96, 'detecte': True/False, ...}
```
Une prédiction sous le seuil de détection (≈ Mw 7,96) signifie « magnitude sous le seuil de détection des PEGS ».
La génération est déterministe (graines fixées dans `src/config.py`) : deux exécutions donnent des fichiers identiques.

## Structure
```
notebooks/01_cadrage_eda.ipynb          analyse des données (axe 3)
notebooks/02_generation_features.ipynb  génération de la base synthétique + caractéristiques (axes 3-4)
notebooks/03_modelisation.ipynb         Dummy / Ridge / XGBoost, validation croisée groupée, optimisation (axe 4)
notebooks/04_evaluation_erreurs.ipynb   test, erreurs par magnitude, plancher de bruit, importance, séismes réels (axes 4-5)
notebooks/05_decision_roi_risques.ipynb décision d'alerte, coûts, seuil, ROI, sensibilité, registre des risques (axes 6, 7, 9)
src/config.py        chemins relatifs, constantes, graines, choix de génération
src/pegs_io.py       lecture des fichiers .h5 (synthétiques, bruit, séismes réels)
src/generate.py      générateur de séismes synthétiques (noyaux de Green + fonction source + bruit réel)
src/features.py      prétraitement et caractéristiques temporelles / spectrales / spatiales
src/dataset.py       construction du tableau une-ligne-par-séisme
src/modeling.py      pipelines des 3 modèles, validation croisée groupée, métriques
src/predict.py       predict_magnitude(signal_pegs) : signal brut -> Mw (+ indicateur de détection)
src/decision.py      hypothèses de coûts (sourcées), repondération Gutenberg-Richter, coûts des politiques, ROI
models/              modèles entraînés (.joblib), model_card.json (hyperparamètres, scores), noise_baseline.json (seuil de détection)
data_processed/      train.csv (4 800), test.csv (1 200), real.csv (4) — régénérables, non versionnés
reports/figures/     figures générées par les notebooks
```

## Résultats
| Modèle | MAE validation croisée (train) | **MAE test** [IC 95 %] | RMSE test | R² test |
|---|---|---|---|---|
| DummyRegressor (moyenne) | 0,621 ± 0,007 | 0,623 [0,603 ; 0,643] | 0,722 | 0,00 |
| Ridge (α = 0,32) | 0,362 ± 0,013 | 0,375 [0,360 ; 0,392] | 0,475 | 0,57 |
| **XGBRegressor optimisé** | **0,281 ± 0,009** | **0,295 [0,282 ; 0,308]** | 0,380 | 0,72 |

- **Plancher de détection** : sur du bruit pur le modèle répond Mw ≈ 7,6 ; sous ≈ 8,0 la prédiction n'est pas fiable.
- **Grandes magnitudes** : 41 % des Mw ≥ 9 sous-estimés de plus de 0,3 (détectabilité / position de la source, effet de bord).
- **Alerte « séisme majeur » (Mw ≥ 8,5)** au seuil 8,3 : rappel 86 %, précision 85 % (sur base synthétique uniforme).
- **Séismes réels** : 1 sur 4 détecté (Chignik 2021 : 8,38 pour 8,15) ; les 3 autres (Mw 7,6–7,9) sont sous le seuil.

## Valeur pour la décision (hypothèses explicites, `src/decision.py`)
Décision : déclencher l'alerte tsunami majeure 5 min après un séisme de Mw ≥ 7 en Alaska (séisme majeur = Mw ≥ 8,5).
- Fréquences réelles (Gutenberg-Richter, b ≈ 1,15) : un séisme majeur tous les ~52 ans.
- **Seuil retenu : Mw prédite ≥ 8,3** → rappel 78 %, précision 51 % (fréquences réelles).
- Coût attendu : 1,60 M$/an contre 3,29 M$/an en attendant la phase W → gain ≈ 1,7 M$/an, ROI an 1 ≈ 1 000 % (espérance).
- **Hypothèse critique** : nombre de victimes évitées par une alerte plus précoce ; rentable dès 3. Scénario pessimiste : ROI −164 %.
- **Recommandation** : déploiement en mode observation (sans pouvoir de décision) avant tout usage opérationnel.

## Sources des hypothèses
- Alaska Earthquake Center, *Earthquake risk* : <https://seismic.alaska.gov/earthquake-risk.html>
- NOAA / ITIC, *Hawaii Tsunami Warnings 1986 & 1994* : <https://www.weather.gov/media/itic-car/hawaii_tsunami_warnings/Hawaii_Tsunami_Warnings_1986_1994_DVD_insert_20131009.pdf>
- Honolulu Star-Bulletin (1999) : <https://archives.starbulletin.com/1999/06/03/news/story8.html>
- US DOT, *Value of a Statistical Life* : <https://www.transportation.gov/office-policy/transportation-policy/revised-departmental-guidance-on-valuation-of-a-statistical-life-in-economic-analysis>
- Q. Bletery, mémoire d'HDR (HAL) ; Licciardi et al. (2022), *Nature* ; Hourcade et al. (2025), *JGR: ML & Computation*

## Usage de l'IA générative
Claude (Anthropic) a été utilisé comme assistant pour : la lecture de la documentation, l'exploration de la structure des fichiers HDF5,
la correction des bugs et des commentaires.

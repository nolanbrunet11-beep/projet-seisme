"""Chemins et constantes du projet (chemins RELATIFS : le dépôt doit tourner sur une machine vierge)."""
import os
from pathlib import Path

# Racine du projet = dossier parent de src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Le dossier data/ du dépôt GitLab PEGSGraph doit être copié ici : <projet>/data/
DATA_DIR = Path(os.environ.get("PEGS_DATA_DIR", PROJECT_ROOT / "data"))   # surchargeable par variable d'environnement
DATASET_DIR = DATA_DIR / "PEGSgraph_dataset"
REAL_DIR = DATA_DIR / "real_time"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"   # dataset de caractéristiques généré (régénérable, non versionné)

# Fichiers fournis par PEGSGraph (Hourcade, Juhel & Bletery, 2025)
KERNEL_FILE = DATASET_DIR / "01_kernel_PEGS_alaska_complete_grid.h5"
PEGS_DB_FILE = DATASET_DIR / "03_PEGS_db_alaska_complete_grid_tiny.h5"
NOISE_FILE = DATASET_DIR / "04_noise_db_alaska_complete_example.h5"
GRAPH_DB_FILE = DATASET_DIR / "graph_db_alaska_Q4_grid_random_complete_tiny.h5"

# Constantes physiques / de traitement
FS = 1.0                 # fréquence d'échantillonnage (Hz) — vérifiée dans le notebook 01
GRAPH_ORIGIN_IDX = 300   # dans les fichiers "graph_db" et "real_time", l'origine du séisme est à l'échantillon 300
RANDOM_STATE = 42        # graine unique pour toute la reproductibilité

# Choix de génération (validés à l'étape 4)
T_DECISION = 300                   # instant de décision : 300 s après l'origine du séisme
MW_RANGE = (7.0, 9.5)              # Mw uniforme ; le modèle de fonction source n'est valable qu'au-dessus de Mw 7
N_TRAIN, N_TEST = 4800, 1200       # 6 000 séismes, 80 % / 20 %
NOISE_TRAIN = list(range(0, 18))   # 18 h de bruit réel réservées à l'entraînement
NOISE_TEST = list(range(18, 24))   # 6 h de bruit réel réservées au test (jamais vues à l'entraînement)
SEED_TRAIN, SEED_TEST = 2026, 7    # graines de génération distinctes

# Palette (cohérente sur toutes les figures)
C_SIGNAL = "#2a78d6"     # PEGS / signal
C_NOISY = "#eb6834"      # signal + bruit
C_MUTED = "#8a8984"      # éléments secondaires

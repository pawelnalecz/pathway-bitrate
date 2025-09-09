# ADD src/ TO PATH
from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

from src import data_manager
from config.local_config import DATA_PATH

# SET DATA DIR

DATA_MANAGER = data_manager.DataManager(DATA_PATH)
WELLS = DATA_MANAGER.wells


# CONFIGURATION

WELLS_SELECTED = WELLS[
    WELLS['experiment'].isin([
        # '2023-08-28-BEAS2B--intensities',
        # '2024-01-08-STE1',
        '2024-07-30-STE1',
        '2024-08-08-STE1',
        # '2024-09-08-BEAS2B',
        # '2024-09-18-BEAS2B-bad',

        '2024-10-22-BEAS2B',
        # '2024-11-20-BEAS2B',
        # '2024-11-20-BEAS2B--first-11h',
        '2024-11-27-BEAS2B',
        # '2024-12-18-BEAS2B',
        '2024-12-23-BEAS2B',
        '2025-02-04-BEAS2B',
    ])
]

SET_TYPES = [
    'main',
    'main-self',
    'main+cell+inh',
    'main+cell+inhtype',
    'exp+inh-self',
    ]


DATASET_IDS = ['ls+cell+inhs']#, 'ls+cell+inhs-q1']
MODEL_IDS = ['nn']
TRAIN_IDS = ['main-q0']
TEST_IDS = ['q1']

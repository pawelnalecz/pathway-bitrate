import numpy as np

# delay in minutes for computation of response strength
RESPONSIVENESS_DELAY = 7

# Tracks quality 
# q0: tracks longer than the following threshold
TRACK_LENGTH_THR = 3 * 60
# q3: tracks in q2 with responsiveness above the following threshold
RESPONSIVENESS_THR = np.log(1.03)

# Imputation
KEEP_START = 10
KEEP_END = 10

# the following values are defaults and can be overriden for each experiment separately in metadata.yaml
DEFAULT_METADATA = {
    # channel names
    'receptor_channel': 'OptoFGFR',
    'reporter_channel': 'ERKKTR',

    # tracks quality
    # q1: tracks in q0 with mean log receptor (standardized with populational avg and std) between the following thresholds
    # 'receptor_lower_thr' # specified in metadata.yaml
    # 'receptor_upper_thr' # specified in metadata.yaml
    # q2: tracks in q1 with mean log reporter (standardized with populational avg and std) between the following thresholds
    'reporter_lower_thr': 0.,
    'reporter_upper_thr': 8.0,
}

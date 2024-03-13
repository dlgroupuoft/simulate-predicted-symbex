import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants.timegroup_constants import *

# ===========================================================================

# Overhead
MODEL_PREDICTION_COST = 0.078 # model prediction time for rand forest
ANALYSIS_SAVE_RESTORE_COST = 5000 # 5 sec: write a lot of mem to disk # 1.5 microsecond
PUSH_POP_SAVE_LIST_COST = 0.0011 # 1.1 microsecond

# Exclude path with less than this many paths / total analysis time
MIN_PATH_THRESHOLD = 100
MIN_TIME_THRESHOLD = 3600 # sec


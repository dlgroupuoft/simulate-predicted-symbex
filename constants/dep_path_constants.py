import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants.timegroup_constants import *

# ===========================================================================

# Avg exec/solving time to use, when the exec/solving time data is not collected
AVG_EXEC_MS = 6.4 
AVG_UNSAT_SOLVE_MS = 1656

# Overhead
AVG_FEAT_COLLECT_MS = 0.324
MODEL_PREDICTION_MS = 1.02 # model prediction time for rand forest # 20 estimators, 20 max depth, sqrt for max feature, no balance training
ANALYSIS_SAVE_RESTORE_MS = 0.0015 # 1.5 microsecond

REMOVE_PARENT_CHILD_RANKLIST_COST = 0.001 # 1 microsecond
LOOKUP_COST = 0.001 # Should be < 1 microsecond, but we assume lookup cost to be larger (1 microsecond)
UPDATE_ONE_ELE_RANK_COST = 0.001
HEAP_PUSH_POP_COST = 0.0011 # 1.1 microsecond (pop takes a bit longer than push, but we use the average)
REHEAPIFY_COST = 0.0188 # reheapify after parent / child are removed

# This should be called when constraint solving finds an unsat state.
# As only child states get popped & got the rank list,
# we remove all child states to stop sym exec if the state
# is unsat.
# Also, the child state may either in the SearchYieldStates.worklist
# or in the ranklist. Discarded states should not be in any of these
# lists. The is_in_list indicate whether the child state should be
# removed from both lists.
REMOVE_UNSAT_CHILD_MAX_DEPTH=500000


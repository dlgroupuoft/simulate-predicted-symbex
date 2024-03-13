
import random

from constants.constants import *

def get_state_hash(num_state_stats_map):
    state_hash = random.randint(1, RANKLIST_HASH_MAX)
    while state_hash in num_state_stats_map:
        state_hash = random.randint(1, RANKLIST_HASH_MAX)
    return state_hash


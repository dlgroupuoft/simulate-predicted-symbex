
import os
import sys

import random
from heapq import heappop, heappush, heapify

from utils.common_ranklist import get_state_hash
from constants.dep_path_constants import *
from constants.constants import *

# ======================================================================

class SearchState():
    state_num = None
    parent_state = None
    child_states = []
    is_in_list = None
    is_in_ranklist = False
    state_depth = None
    sat_rank = None
    list_stay_iter = 0
    list_stay_pop = 0
    def __init__(self, state_num, parent_state, state_depth):
        self.state_num = state_num
        self.parent_state = parent_state
        self.state_depth = state_depth
        self.child_states = []

    def update_sat_rank(self, sat_rank):
        self.sat_rank = sat_rank

    def add_child(self, child_state):
        self.child_states.append(child_state)

    def increment_liststay_pop(self):
        self.list_stay_pop += 1

    def increment_liststay_iter(self):
        self.list_stay_iter += 1

def remove_unsat_child_state_with_depth(state, depth):
    if depth == REMOVE_UNSAT_CHILD_MAX_DEPTH:
        return 0

    num_states_removed = 0
    num_states_visited = 0
    for child_state in state.child_states:
        num_states_visited += 1
        if child_state.is_in_list == False:
            continue
        child_state.is_in_list = False
        num_states_removed += 1

        additional_states_removed, additional_states_visited = remove_unsat_child_state_with_depth(child_state, depth + 1)
        num_states_removed += additional_states_removed
        num_states_visited += additional_states_visited
    state.child_states = []
    return num_states_removed, num_states_visited

def remove_unsat_child_state(state):
    return remove_unsat_child_state_with_depth(state, 0)

# ======================================================================

class SearchYieldStates():
    worklist = []
    start_states = []
    blk_parent_to_child = dict()
    blk_child_to_parent = dict()

    def __init__(self, start_states, state_pred_dict,
                 blk_parent_to_child, blk_child_to_parent):
        self.start_states = start_states
        self.blk_parent_to_child = blk_parent_to_child
        self.blk_child_to_parent = blk_child_to_parent

    def add_to_list(self, state_num, parent_state, state_depth):
        child_state = SearchState(state_num, parent_state, state_depth)
        child_state.is_in_list = True
        if parent_state is not None:
            parent_state.add_child(child_state)
        self.worklist.append(child_state)

    def pop_list(self):
        state = self.worklist.pop(0)
        return state

    def search_yield_states(self):
        self.worklist = []
        for start_state in self.start_states:
            self.add_to_list(start_state, None, 0)
        while len(self.worklist) > 0:
            curr_state = self.pop_list()
            if curr_state.is_in_list == False:
                continue

            if curr_state.state_num in self.blk_parent_to_child:
                next_state_nums = self.blk_parent_to_child[curr_state.state_num]
                for next_state_num in next_state_nums:
                    self.add_to_list(next_state_num, curr_state, curr_state.state_depth + 1)
            yield curr_state

# ======================================================================

def push_ranklist_common(ranklist, num_state_stats_map, rank, state_stats,
                         use_pred_rank, rank_early_states_first):
    assert(use_pred_rank is not None) # either True or False
    if rank_early_states_first:
        state = state_stats[0]
        if type(state) is int:
            state_num = state
        else:
            state_num = state.state_num
        state_hash = state_num
    else:
        state_hash = get_state_hash(num_state_stats_map)
    num_state_stats_map[state_hash] = state_stats

    # Use -rank, so we have a max heap (prioritize large value)
    rank_data = (-rank, state_hash)
    if use_pred_rank:
        heappush(ranklist, rank_data)
    else:
        ranklist.append(rank_data)

def push_ranklist_rank_listsize(ranklist, num_state_stats_map, rank, state_stats, use_pred_rank, rank_early_states_first):
    # state_stats: [state, wait_finished]
    state = state_stats[0]
    state.is_in_list = True
    state.is_in_ranklist = True

    push_ranklist_common(ranklist, num_state_stats_map, rank, state_stats, use_pred_rank, rank_early_states_first)

def pop_ranklist_once(ranklist, num_state_stats_map, use_pred_rank):
    assert(len(ranklist) > 0)
    assert(use_pred_rank is not None) # either True or False
    if use_pred_rank:
        rank, state_hash = heappop(ranklist)
    else:
        rand_ind = random.randint(0, len(ranklist) - 1)
        rank, state_hash = ranklist.pop(rand_ind)
    assert(state_hash in num_state_stats_map)
    state_stats = num_state_stats_map[state_hash]
    del num_state_stats_map[state_hash]
    return rank, state_stats

# For rank list size
def reheapify_rank_list(ranklist, num_state_stats_map):
    i = 0
    while i < len(ranklist):
        rank, state_hash = ranklist[i]
        assert(state_hash in num_state_stats_map)
        state_stats = num_state_stats_map[state_hash]
        state = state_stats[0]
        if state.is_in_list == False:
            ranklist.pop(i)
            del num_state_stats_map[state_hash]
        else:
            i += 1
    heapify(ranklist)

#===============================================================================

def get_sat_unsat_depth_rank(sat_rank, depth,
                             rank_reverse_unsat_ratio,
                             rank_reverse_depth_minuend):
    assert(rank_reverse_unsat_ratio is not None)
    if rank_reverse_depth_minuend is None:
        unsat_neg_depth = 1
    else:
        unsat_neg_depth = rank_reverse_depth_minuend - depth
    return (sat_rank * depth) + rank_reverse_unsat_ratio * (1 - sat_rank) * unsat_neg_depth

def reheapify_ranklist_reverse_unsat(ranklist, num_state_stats_map,
                                     rank_reverse_unsat_ratio, rank_reverse_depth_minuend):
    if len(ranklist) == 0:
        return

    min_depth = None
    for _old_rank, state_hash in ranklist:
        state, solved = num_state_stats_map[state_hash]
        if min_depth is None:
            min_depth = state.state_depth
        else:
            min_depth = min(min_depth, state.state_depth)
        assert(state.sat_rank is not None)
        assert(state.sat_rank >= 0 and state.sat_rank <= 1)
    assert(min_depth is not None)

    for i in range(len(ranklist)):
        _old_rank, state_hash = ranklist[i]
        state, solved = num_state_stats_map[state_hash]

        curr_state_depth = state.state_depth - min_depth
        sat_unsat_depth_rank = get_sat_unsat_depth_rank(state.sat_rank, curr_state_depth,
                                    rank_reverse_unsat_ratio=rank_reverse_unsat_ratio,
                                    rank_reverse_depth_minuend=rank_reverse_depth_minuend)

        # Use -ve rank: we have a max heap
        ranklist[i] = (-sat_unsat_depth_rank, state_hash)
    heapify(ranklist)

#===============================================================================

def curr_state_time_class_confidence(state_pred_dict, curr_state):
    actual_class, sat_confidence, constr_solve_time, execute_time = state_pred_dict[curr_state]
    if execute_time is None:
        execute_time = AVG_EXEC_MS

    # This happens for unsat paths in keep_unsat, where we don't solve
    if constr_solve_time is None:
        assert(not actual_class)
        constr_solve_time = AVG_UNSAT_SOLVE_MS
    return actual_class, sat_confidence, constr_solve_time, execute_time

def get_prev_sat_diff_score(args, state, stats, state_pred_dict, rank_prev_back, rank_prev_sat_diff_ratio):
    assert(rank_prev_back is not None)
    assert(rank_prev_sat_diff_ratio is not None)
    assert(rank_prev_back > 0)

    assert(state.sat_rank is not None)
    curr_state = state
    prev_state = None
    for i in range(rank_prev_back):
        prev_state = curr_state.parent_state
        if prev_state is None:
            break
        assert(prev_state.sat_rank is not None)

    if prev_state is None:
        prev_sat_diff = 0
    else:
        if prev_state.sat_rank is not None:
            prev_sat_rank = prev_state.sat_rank
        else:
            # This prev state is filtered and solved; Need to predict again
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, prev_state.state_num)
            stats.add_strategy_overhead(MODEL_PREDICTION_MS)
            prev_sat_rank = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)

        prev_sat_diff = max(prev_sat_rank - state.sat_rank, 0)
    return rank_prev_sat_diff_ratio * prev_sat_diff + (1 - state.sat_rank)



import pickle
import os
import random
import re
import sys
from copy import copy, deepcopy
from heapq import heappop, heappush, heapify

#===============================================================================

from constants.constants import *
from constants.dep_path_constants import *
from utils.args import parse_args, parse_wait_filter
from utils.common import safe_pickle, load_pickle_res, get_tenth_time_ind, apprx_equal
from utils.TimeBudgetStats import TimeBudgetStats, TotalStats
from utils.time_stats import *
from utils.dep_ranklist import *
from utils.simulate_model import *

#===============================================================================

def get_removed_dep_total_stats(state_pred_dict):
    total_stats = TotalStats()
    for curr_state in state_pred_dict.keys():
        actual_class, _sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state)
        total_stats.update(actual_class, constr_solve_time, execute_time)
        raw_constr_time = state_pred_dict[curr_state][2]
        assert((raw_constr_time is not None) or (not actual_class))
    return total_stats

def get_total_stats(state_pred_dict, blk_parent_to_child):
    total_stats = TotalStats()
    curr_states = blk_parent_to_child[None]
    unique_states = set()
    while len(curr_states) > 0:
        next_states = []
        for curr_state in curr_states:
            assert(curr_state not in unique_states)
            unique_states.add(curr_state)

            actual_class, _sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state)
            total_stats.update(actual_class, constr_solve_time, execute_time)
            raw_constr_time = state_pred_dict[curr_state][2]
            if raw_constr_time is None and actual_class:
                assert(raw_constr_time is not None)

            if actual_class and curr_state in blk_parent_to_child:
                for next_state in blk_parent_to_child[curr_state]:
                    next_states.append(next_state)

        curr_states = next_states
    return total_stats

#===============================================================================

def no_or_hard_pred_dfs_explore(args, start_states,
                                state_pred_dict, blk_parent_to_child,
                                progress_percent_timebudget, percent_timebudget_ind,
                                progress_percent_satpath, percent_satpath_ind,
                                stats, total_stats):
    def no_pred_wait_analyze(actual_class, constr_solve_time, sat_confidence):
        skipped = False
        pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
        if stats.wait_time > 0:
            wait_finished = stats.perform_wait(actual_class, constr_solve_time, pred_sat_val) 
            if args.hard_pred_skip_unknown and not wait_finished:
                stats.skip(actual_class, constr_solve_time, pred_sat_val)
                skipped = True
        else:
            stats.add_analysis_time(actual_class, constr_solve_time, pred_sat_val)
        return skipped

    def hard_pred_wait_analyze(actual_class, constr_solve_time, sat_confidence):
        skipped = False
        pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
        if stats.wait_time > 0:
            wait_finished = stats.perform_wait(actual_class, constr_solve_time, pred_sat_val)
        else:
            wait_finished = False

        if not wait_finished:
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
            predict_analyze = (pred_sat_val >= args.hard_pred_confidence_threshold)
            stats.add_strategy_overhead(MODEL_PREDICTION_MS)
            if predict_analyze:
                # Keep waiting, the wait unfinished time is part of the analysis time
                stats.long_analyze(actual_class, constr_solve_time, pred_sat_val)
                stats.use_wait_not_finished(actual_class)
            else:
                stats.skip(actual_class, constr_solve_time, pred_sat_val)
                skipped = True
            stats.count_timegroup_sat_unsat_pred(constr_solve_time, actual_class, predict_analyze)
        stats.check_wait_consistent()
        return skipped

    worklist = []
    curr_states = start_states
    while len(worklist) > 0 or len(curr_states) > 0:
        for curr_state in curr_states:
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state)
            stats.add_find_time(execute_time, actual_class)

            if args.mode == "NOPRED":
                skipped = no_pred_wait_analyze(actual_class, constr_solve_time, sat_confidence)
            else:
                # For hard pred, no matter we predict or not (i.e. timeout or not)
                # we still require feat collection, so we can gradually build
                # features for successor states
                stats.add_strategy_overhead(AVG_FEAT_COLLECT_MS)
                skipped = hard_pred_wait_analyze(actual_class, constr_solve_time, sat_confidence)

            if not skipped:
                worklist.append(curr_state)
            stats.update_worklist_size(len(worklist))
            percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
            percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

        stats.update_worklist_size(len(worklist))
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)
        if len(worklist) == 0:
            break

        parent_state = worklist.pop(0)
        actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, parent_state)
        if actual_class and parent_state in blk_parent_to_child:
            curr_states = blk_parent_to_child[parent_state]
        else:
            curr_states = []
    assert(len(worklist) == 0)
    return percent_timebudget_ind, percent_satpath_ind

# ======================================================================

def removed_dep_rank_explore(args, state_pred_dict,
                             progress_percent_timebudget, percent_timebudget_ind,
                             progress_percent_satpath, percent_satpath_ind,
                             stats, total_stats):
    worklist = []
    num_state_stats_map = dict()

    states = [curr_state for curr_state in state_pred_dict.keys()]
    random.shuffle(states)
    state_ind = 0

    while state_ind < len(states) or len(worklist) > 0:
        if state_ind < len(states):
            curr_state = states[state_ind]
            state_ind += 1
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state)
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
            stats.add_find_time(execute_time, actual_class)
            stats.add_strategy_overhead(AVG_FEAT_COLLECT_MS)

            solved = False
            add_model_pred_cost = False
            if stats.wait_time > 0:
                wait_finished = stats.perform_wait(actual_class, constr_solve_time, pred_sat_val)
                solved = wait_finished
            else:
                filter_passed = stats.perform_filter_analyze(pred_sat_val, actual_class, constr_solve_time)
                add_model_pred_cost = (stats.pred_filter is not None)
                solved = filter_passed

            #
            state_res = [curr_state, solved]
            if not solved:
                pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)

                if args.rank_objective == 'random':
                    rank_val = random.random()
                else:
                    rank_val = pred_sat_val
                    add_model_pred_cost = True
                push_ranklist_common(worklist, num_state_stats_map, rank_val, state_res, args.rank_objective != 'random', args.rankEarlyStateFirst)
                stats.count_timegroup_sat_unsat_pred(constr_solve_time, actual_class, pred_sat_val >= 0.5)
                stats.add_strategy_overhead(HEAP_PUSH_POP_COST)

                if stats.wait_time > 0:
                    stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)

            #
            if add_model_pred_cost:
                stats.add_strategy_overhead(MODEL_PREDICTION_MS)

            #
            stats.update_worklist_size(len(worklist))
            percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
            percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

        #
        if len(worklist) > 0 and (len(worklist) >= args.list_size or state_ind >= len(states)):
            _confidence, state_res = pop_ranklist_once(worklist, num_state_stats_map, args.rank_objective != 'random')
            stats.add_strategy_overhead(HEAP_PUSH_POP_COST)

            _confidence = - _confidence
            state, solved = state_res
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, state)
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)

            if not solved:
                # Restore path analysis
                if stats.wait_time > 0:
                    stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)

                assert(constr_solve_time >= stats.wait_time)
                stats.long_analyze(actual_class, constr_solve_time, pred_sat_val)
                stats.use_wait_not_finished(actual_class)

            #
            stats.update_worklist_size(len(worklist))
            percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
            percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

        #
        stats.check_wait_consistent()
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    assert(len(worklist) == 0)
    return percent_timebudget_ind, percent_satpath_ind

# ======================================================================

def unlimited_pred_or_random_rank_explore(args, start_states,
                                          state_pred_dict, blk_parent_to_child,
                                          progress_percent_timebudget, percent_timebudget_ind,
                                          progress_percent_satpath, percent_satpath_ind,
                                          stats, total_stats):
    wait_time = stats.wait_time
    worklist = []
    num_state_stats_map = dict()

    curr_states = start_states
    while len(worklist) > 0 or len(curr_states) > 0:
        next_states = []
        for curr_state in curr_states:
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state)
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
            stats.add_find_time(execute_time, actual_class)
            stats.add_strategy_overhead(AVG_FEAT_COLLECT_MS)

            solved = False
            add_model_pred_cost = False
            if wait_time > 0:
                wait_finished = stats.perform_wait(actual_class, constr_solve_time, pred_sat_val)
                solved = wait_finished
            else:
                filter_passed = stats.perform_filter_analyze(pred_sat_val, actual_class, constr_solve_time)
                add_model_pred_cost = (stats.pred_filter is not None)
                solved = filter_passed

            #
            state_res = [curr_state, solved]
            if not solved:
                pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
                if args.rank_objective == 'random':
                    rank_val = random.random()
                else:
                    rank_val = pred_sat_val
                    add_model_pred_cost = True
                push_ranklist_common(worklist, num_state_stats_map, rank_val, state_res, args.rank_objective != 'random', args.rankEarlyStateFirst)
                stats.count_timegroup_sat_unsat_pred(constr_solve_time, actual_class, pred_sat_val >= 0.5)
                stats.add_strategy_overhead(HEAP_PUSH_POP_COST)
                if wait_time > 0:
                    stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)
            ######
            else:
                # Solved, we know it's sat or not
                # So, don't need prediction or save/restore states any more
                # But we need to pull the state from the list to get its successors
                if actual_class:
                    # We know it's set: so set sat confidence to 1
                    push_ranklist_common(worklist, num_state_stats_map, 1, state_res, args.rank_objective != 'random', args.rankEarlyStateFirst)
                    stats.add_strategy_overhead(HEAP_PUSH_POP_COST)
            ######
            # elif curr_state in blk_parent_to_child:
            #     for next_state in blk_parent_to_child[curr_state]:
            #         next_states.append(next_state)
            ######

            #
            if add_model_pred_cost:
                stats.add_strategy_overhead(MODEL_PREDICTION_MS)

            #
            stats.update_worklist_size(len(worklist))
            percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
            percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

        stats.update_worklist_size(len(worklist))
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

        #
        if len(worklist) > 0:
            _confidence, state_res = pop_ranklist_once(worklist, num_state_stats_map, args.rank_objective != 'random')
            stats.add_strategy_overhead(HEAP_PUSH_POP_COST)

            _confidence = - _confidence
            state, solved = state_res
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, state)
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)

            if not solved:
                # Restore path analysis
                if wait_time > 0:
                    stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)

                assert(constr_solve_time >= wait_time)
                stats.long_analyze(actual_class, constr_solve_time, pred_sat_val)
                stats.use_wait_not_finished(actual_class)

            #
            if actual_class and state in blk_parent_to_child:
                for next_state in blk_parent_to_child[state]:
                    next_states.append(next_state)

        #
        curr_states = next_states

        #
        stats.check_wait_consistent()
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    assert(len(worklist) == 0)
    return percent_timebudget_ind, percent_satpath_ind

#===============================================================================


def listsize_pred_or_random_rank_explore(args, start_states,
                                         state_pred_dict,
                                         blk_parent_to_child, blk_child_to_parent,
                                         progress_percent_timebudget, percent_timebudget_ind,
                                         progress_percent_satpath, percent_satpath_ind,
                                         stats, total_stats):
    ranklist = []
    num_state_stats_map = dict()
    stats.ranklist_path_depth_sum = 0
    stats.ranklist_num_paths = 0

    def remove_ranklist_sat_parent(curr_state):
        num_removed_states = 0
        removed_constr_solve_time = 0
        while curr_state:
            if curr_state.is_in_list:
                curr_state.is_in_list = False
                num_removed_states += 1

                actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, curr_state.state_num)

                # Has timeout: omitted analysis time does not include wait time
                # is_in_ranklist: passed through the path finding stage (actually waited)
                if stats.wait_time > 0 and curr_state.is_in_ranklist:
                    assert(constr_solve_time >= stats.wait_time)
                    removed_constr_solve_time += (constr_solve_time - stats.wait_time)
                else:
                    removed_constr_solve_time += constr_solve_time
            else:
                break

            parent_state = curr_state.parent_state
            curr_state.parent_state = None
            curr_state = parent_state
        return num_removed_states, removed_constr_solve_time

    #
    def known_result_remove_parent_child(stats, state, actual_class, need_reheapify=True):
        if args.no_path_prune:
            return
        if not actual_class:
            num_states_removed, num_states_visited = remove_unsat_child_state(state)
            stats.num_unsat_omitted += num_states_removed
        else:
            num_states_removed, removed_constr_solve_time = remove_ranklist_sat_parent(state.parent_state)
            num_states_visited = num_states_removed

            stats.num_sat_omitted += num_states_removed
            stats.total_time_sat_analysis_omitted += removed_constr_solve_time
            state.parent_state = None
        state.is_in_list = False
        state.is_in_ranklist = False
        stats.remove_parent_child_listsize_predrank(num_states_visited, REMOVE_PARENT_CHILD_RANKLIST_COST)

        if need_reheapify:
            reheapify_rank_list(ranklist, num_state_stats_map)
            stats.add_strategy_overhead(REHEAPIFY_COST)

    def increase_all_liststay_pop():
        for rank, state_hash in ranklist:
            state, solved = num_state_stats_map[state_hash]
            state.increment_liststay_pop()

    def increase_all_liststay_iter():
        for rank, state_hash in ranklist:
            state, solved = num_state_stats_map[state_hash]
            state.increment_liststay_iter()

    def pop_analyze():
        _sat_rank_val, state_res = pop_ranklist_once(ranklist, num_state_stats_map, args.rank_objective != 'random')
        increase_all_liststay_pop()
        stats.add_strategy_overhead(HEAP_PUSH_POP_COST)
        state, solved = state_res
        stats.ranklist_path_depth_sum -= state.state_depth
        stats.ranklist_num_paths -= 1

        def analyze_state_use_wait(state_pred_dict, state):
            actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, state.state_num)
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
            if stats.wait_time > 0:
                stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)
            assert(constr_solve_time >= stats.wait_time)
            stats.long_analyze(actual_class, constr_solve_time, pred_sat_val)
            stats.use_wait_not_finished(actual_class)
            return actual_class

        def analyze_remove_parent_state(parent_state):
            if parent_state and parent_state.is_in_list:
                parent_state.is_in_list = False
                parent_actual_class = analyze_state_use_wait(state_pred_dict, parent_state)
                known_result_remove_parent_child(stats, parent_state, parent_actual_class, need_reheapify=False)
                return parent_actual_class
            elif parent_state:
                # Already analyzed
                parent_actual_class, _sat_confidence, _constr_solve_time, _execute_time = curr_state_time_class_confidence(state_pred_dict, parent_state.state_num)
                stats.add_strategy_overhead(LOOKUP_COST)
                return parent_actual_class
            else:
                return None

        actual_class = analyze_state_use_wait(state_pred_dict, state)
        stats.pop_state(state, actual_class)

        # Now we finally know it's sat or unsat
        if args.rankPrevBack is None or args.rankNoAnalyzePrev:
            known_result_remove_parent_child(stats, state, actual_class)
        else:
            assert(args.rankPrevSatDiffRatio is not None)
            assert(args.rank_objective == 'hybrid')
            known_result_remove_parent_child(stats, state, actual_class, need_reheapify=False)

            # When using the args.rankPrevBack score:
            # If current state is unsat, remove its potentially sat parent
            if not actual_class:
                if not args.rankClearWindowUnsatParent:
                    analyze_remove_parent_state(state.parent_state)
                else:
                    curr_state = state
                    max_parent_state = None
                    for i in range(args.rankPrevBack):
                        if curr_state.parent_state is None:
                            break
                        max_parent_state = curr_state.parent_state
                        curr_state = max_parent_state
                    max_parent_is_sat = analyze_remove_parent_state(max_parent_state)
                    if max_parent_is_sat:
                        max_ind = i - 2 # Already know satisfiability of first & last state: -2
                        min_ind = 0
                        start_state = state.parent_state
                        while max_ind >= min_ind:
                            avg_ind = (max_ind + min_ind) // 2
                            curr_parent_state = start_state
                            for i in range(avg_ind):
                                curr_parent_state = start_state.parent_state
                            assert(curr_parent_state)
                            stats.add_strategy_overhead(i * LOOKUP_COST)

                            parent_is_sat = analyze_remove_parent_state(curr_parent_state)
                            if parent_is_sat:
                                max_ind = avg_ind - 1
                            else:
                                min_ind = avg_ind + 1
            reheapify_rank_list(ranklist, num_state_stats_map)
            stats.add_strategy_overhead(REHEAPIFY_COST)

    def save_timebudget_stats(stats, percent_timebudget_ind, percent_satpath_ind):
        stats.update_worklist_size(len(ranklist))
        stats.check_wait_consistent()
        ret_percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        ret_percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)
        return ret_percent_timebudget_ind, ret_percent_satpath_ind

    #
    yield_states_obj = SearchYieldStates(start_states, state_pred_dict,
                                         blk_parent_to_child, blk_child_to_parent)
    for state in yield_states_obj.search_yield_states():
        # Return object here; For all other strategies, we say state, it actually refers to state num
        actual_class, sat_confidence, constr_solve_time, execute_time = curr_state_time_class_confidence(state_pred_dict, state.state_num)
        pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
        stats.add_find_time(execute_time, actual_class)
        if args.rank_objective != 'random':
            stats.add_strategy_overhead(AVG_FEAT_COLLECT_MS)
        solved = False
        add_model_pred_cost = False
        if stats.wait_time > 0:
            wait_finished = stats.perform_wait(actual_class, constr_solve_time, pred_sat_val)
            solved = wait_finished
        else:
            filter_passed = stats.perform_filter_analyze(pred_sat_val, actual_class, constr_solve_time, state_depth=state.state_depth)
            add_model_pred_cost = (stats.pred_filter is not None)
            solved = filter_passed

        if solved:
            known_result_remove_parent_child(stats, state, actual_class)
        else:
            state_stats = [state, solved]
            pred_sat_val = simulate_pred_sat_rank(args, actual_class, sat_confidence, constr_solve_time)
            state.update_sat_rank(pred_sat_val)
            stats.count_timegroup_sat_unsat_pred(constr_solve_time, actual_class, pred_sat_val >= 0.5)

            if args.rank_objective != 'hybrid':
                assert(args.rankPrevBack is None)
                assert(args.rankPrevSatDiffRatio is None)
                if args.rankUnsatRatio is not None:
                    assert(pred_sat_val == state.sat_rank)
                    sat_unsat_depth_rank = get_sat_unsat_depth_rank(state.sat_rank, state.state_depth, args.rankUnsatRatio, args.rankDepthMinuend)
                    push_ranklist_rank_listsize(ranklist, num_state_stats_map, sat_unsat_depth_rank, state_stats, args.rank_objective != 'random', args.rankEarlyStateFirst)

                    if not args.rankDepthReset:
                        stats.add_strategy_overhead(HEAP_PUSH_POP_COST)
                    else:
                        # For convenience, here we push first then heapify
                        # With actual implementation, should add to list first & heapify altogether
                        reheapify_ranklist_reverse_unsat(ranklist, num_state_stats_map, args.rankUnsatRatio, args.rankDepthMinuend)
                        stats.add_strategy_overhead(len(ranklist) * UPDATE_ONE_ELE_RANK_COST)
                        stats.add_strategy_overhead(REHEAPIFY_COST)
                else:
                    if args.rank_objective == 'unsat':
                        # unsat rank: use unsat likelihood instead of sat likelihood
                        use_pred_sat_val = 1 - pred_sat_val
                    else:
                        use_pred_sat_val = pred_sat_val
                    push_ranklist_rank_listsize(ranklist, num_state_stats_map, use_pred_sat_val, state_stats, args.rank_objective != 'random', args.rankEarlyStateFirst)
                    stats.add_strategy_overhead(HEAP_PUSH_POP_COST)
            else:
                assert(args.rankPrevBack is not None)
                assert(args.rankPrevSatDiffRatio is not None)
                prev_sat_diff_score = get_prev_sat_diff_score(args, state, stats, state_pred_dict, args.rankPrevBack, args.rankPrevSatDiffRatio)
                push_ranklist_rank_listsize(ranklist, num_state_stats_map, prev_sat_diff_score, state_stats, args.rank_objective != 'random', args.rankEarlyStateFirst)
                stats.add_strategy_overhead(HEAP_PUSH_POP_COST)

            stats.ranklist_path_depth_sum += state.state_depth
            stats.ranklist_num_paths += 1
            assert(len(ranklist) <= args.list_size)

            #
            if args.rank_objective != 'random':
                add_model_pred_cost = True
            if stats.wait_time > 0:
                stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_MS)

            #
            increase_all_liststay_iter()
            if len(ranklist) == args.list_size:
                pop_analyze()

        #
        if add_model_pred_cost:
            stats.add_strategy_overhead(MODEL_PREDICTION_MS)

        percent_timebudget_ind, percent_satpath_ind = save_timebudget_stats(stats, percent_timebudget_ind, percent_satpath_ind)

    #
    while len(ranklist) > 0:
        increase_all_liststay_iter()
        pop_analyze()
        percent_timebudget_ind, percent_satpath_ind = save_timebudget_stats(stats, percent_timebudget_ind, percent_satpath_ind)

    return percent_timebudget_ind, percent_satpath_ind

#===============================================================================

def get_start_states(blk_parent_to_child):
    start_list = copy(blk_parent_to_child[None])
    while len(start_list) > 0:
        curr_start = []
        min_state = min(start_list)

        for i in range(0, 3):
            start_state = min_state + i
            if start_state in start_list:
                start_list.remove(start_state)
                curr_start.append(start_state)
            else:
                break
        assert(len(curr_start) > 0)
        yield curr_start

def guide_analyze(args,
                  state_pred_dict,
                  blk_parent_to_child, blk_child_to_parent,
                  wait_filter):

    #
    progress_percent_timebudget = dict()
    progress_percent_satpath = dict()
    percent_timebudget_ind = percent_satpath_ind = 0
    stats = TimeBudgetStats(wait_filter)
    if args.mode == "REMOVEDEP_LISTSIZE":
        total_stats = get_removed_dep_total_stats(state_pred_dict)
    else:
        total_stats = get_total_stats(state_pred_dict, blk_parent_to_child)
    if total_stats.num_total_sat_paths == 0:
        return None

    def explore(curr_start, percent_timebudget_ind, percent_satpath_ind):
        if args.mode == "NOPRED" or args.mode == "HARDPRED":
            percent_timebudget_ind, percent_satpath_ind = no_or_hard_pred_dfs_explore(
                    args, curr_start,
                    state_pred_dict, blk_parent_to_child,
                    progress_percent_timebudget, percent_timebudget_ind,
                    progress_percent_satpath, percent_satpath_ind,
                    stats, total_stats)
        elif args.mode == "INDEP_RANK":
            percent_timebudget_ind, percent_satpath_ind = unlimited_pred_or_random_rank_explore(
                    args, curr_start,
                    state_pred_dict, blk_parent_to_child,
                    progress_percent_timebudget, percent_timebudget_ind,
                    progress_percent_satpath, percent_satpath_ind,
                    stats, total_stats)
        elif args.mode == "DEP_RANK":
            percent_timebudget_ind, percent_satpath_ind = listsize_pred_or_random_rank_explore(
                                         args, curr_start,
                                         state_pred_dict,
                                         blk_parent_to_child, blk_child_to_parent,
                                         progress_percent_timebudget, percent_timebudget_ind,
                                         progress_percent_satpath, percent_satpath_ind,
                                         stats, total_stats)
        else:
            assert(False)
        return percent_timebudget_ind, percent_satpath_ind

    if args.mode == "REMOVEDEP_LISTSIZE":
        percent_timebudget_ind, percent_satpath_ind = removed_dep_rank_explore(args,
                             state_pred_dict,
                             progress_percent_timebudget, percent_timebudget_ind,
                             progress_percent_satpath, percent_satpath_ind,
                             stats, total_stats)
    else:
        if not args.broad_start:
            for curr_start in get_start_states(blk_parent_to_child):
                percent_timebudget_ind, percent_satpath_ind = explore(curr_start, percent_timebudget_ind, percent_satpath_ind)
        else:
            start_state_set = set()
            for curr_start in get_start_states(blk_parent_to_child):
                for start_state in curr_start:
                    start_state_set.add(start_state)

            percent_timebudget_ind = explore(start_state_set, percent_timebudget_ind, percent_satpath_ind)

    #
    if args.mode == "NOPRED":
        if wait_filter is not None:
            random_filter, pred_filter, depth_filter, wait_time = wait_filter
            assert(random_filter is None)
            assert(pred_filter is None)
            assert(depth_filter is None)
            has_skipped_paths = (wait_time > 0)
        else:
            has_skipped_paths = False
    elif args.mode == "HARDPRED":
        has_skipped_paths = True
    elif args.mode == "INDEP_RANK": # unlimited path set size
        has_skipped_paths = False
    else:
        assert(args.mode == "DEP_RANK" or args.mode == "REMOVEDEP_LISTSIZE")
        has_skipped_paths = True

    #
    if not has_skipped_paths:
        stats.check_final_analysis_path(total_stats.num_total_paths)
        stats.check_final_finish_notfinish()

    if args.list_size is not None and args.list_size > 0 and args.mode != "REMOVEDEP_LISTSIZE":
        stats.check_final_analysis_time(total_stats.total_analysis_time, total_stats.total_find_time, has_skipped_paths=has_skipped_paths)

    if args.mode == 'INDEP_RANK': # No path pruning for indep ranking
        stats.check_all_notfinished_is_used()

    assert(stats.worklist_curr_size == 0 or stats.worklist_curr_size == 1)
    update_final_budget_stats(progress_percent_timebudget, stats, total_stats, is_percent=True)
    update_final_budget_stats(progress_percent_satpath, stats, total_stats, is_path_percent=True)
    return progress_percent_timebudget, progress_percent_satpath, total_stats

#===============================================================================

def guide_states_in_dir(args):
    app_progress_time_budget = dict()
    wait_filter = parse_wait_filter(args.wait_filter_str, has_dep=True)
    for root, dirs, files in os.walk(args.preds_dir):
        for preds_filename in files:
            preds_filename = os.path.join(root, preds_filename)
            rel_root_dir = os.path.relpath(os.path.dirname(preds_filename), args.preds_dir)

            basename = os.path.basename(preds_filename)
            pred_name_match = re.match(r"(.*)_fold\d+.p", basename)
            if not pred_name_match:
                continue
            basename = pred_name_match.group(1)

            state_cfg_time_filename = os.path.join(args.state_cfg_time_dir, rel_root_dir, basename + ".p")
            if not os.path.isfile(state_cfg_time_filename):
                continue

            #
            with open(preds_filename, "rb") as in_file:
                state_pred_dict = pickle.load(in_file)
            with open(state_cfg_time_filename, "rb") as in_file:
                cfg_child_to_parent, cfg_parent_to_child, blk_child_to_parent, blk_parent_to_child, state_constr_solve_time, state_execute_time = pickle.load(in_file)
            del cfg_child_to_parent, cfg_parent_to_child
            del state_constr_solve_time, state_execute_time

            name_sig = os.path.join(rel_root_dir, basename)
            progress_time_stats = guide_analyze(args, state_pred_dict, blk_parent_to_child, blk_child_to_parent, wait_filter)
            if progress_time_stats is None:
                continue
            app_progress_time_budget[name_sig] = progress_time_stats
            print(".", end="", flush=True)

    safe_pickle(app_progress_time_budget, args.out_filename)
    print()
    print()
    print("End guide_states_in_dir.")

#===============================================================================

if __name__ == "__main__":
    args = parse_args(has_path_dep=True)
    if os.path.isfile(args.out_filename):
        print("{0} already exists; Skipping...".format(args.out_filename))
        sys.exit()

    out_dir = os.path.dirname(args.out_filename)
    if len(out_dir) > 0 and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    guide_states_in_dir(args)


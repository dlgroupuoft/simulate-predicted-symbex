
import os
import pickle
import random
import re
import sys
from copy import copy
from heapq import heappop, heappush, heapify

#===============================================================================

from constants.constants import *
from constants.no_dep_path_constants import *
from utils.args import parse_args, parse_wait_filter
from utils.common import safe_pickle, load_pickle_res, get_tenth_time_ind, apprx_equal
from utils.common_ranklist import get_state_hash
from utils.TimeBudgetStats import TimeBudgetStats, TotalStats
from utils.time_stats import *
from utils.simulate_model import *

#===============================================================================

def path_time_pred(path_info_list, pred_list):
    for i, info_entry in enumerate(path_info_list):
        satisfiable, _num_dependencies, _path_length, path_analysis_time, path_find_time = info_entry
        actual_class, predicted_class, pred_val = pred_list[i]
        assert(satisfiable == actual_class)
        yield actual_class, predicted_class, pred_val, path_analysis_time, path_find_time

def get_sat_rank(predicted_class, pred_val):
    assert(pred_val >= 0.5)
    if predicted_class:
        return pred_val
    else:
        return 1 - pred_val

def get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val):
    sat_rank = get_sat_rank(predicted_class, pred_val)
    sat_rank = simulate_confidence_shift(args, actual_class, sat_rank)
    return sat_rank

#===============================================================================

def get_info_filtered_apps(app_info_dict, app_path_time, app_pred_dict):
    for app_name, path_info_list in app_info_dict.items():
        if len(path_info_list) < MIN_PATH_THRESHOLD:
            continue

        total_path_analysis_time = 0
        for info_entry in path_info_list:
            satisfiable, _num_dependencies, _path_length, path_analysis_time, path_find_time = info_entry
            total_path_analysis_time += (path_analysis_time / 1000) # ms -> sec
        if total_path_analysis_time < MIN_TIME_THRESHOLD:
            continue

        assert(app_name in app_path_time)
        path_time_list = app_path_time[app_name]

        assert(app_name in app_pred_dict)
        pred_list = app_pred_dict[app_name]

        assert(len(path_info_list) == len(pred_list))
        assert(len(path_info_list) == len(path_time_list))

        yield app_name, path_info_list, path_time_list, pred_list

# ===========================================================================

def get_total_stats(path_info_list, pred_list):
    total_stats = TotalStats()
    for actual_class, predicted_class, _pred_val, path_analysis_time, path_find_time in path_time_pred(path_info_list, pred_list):
        total_stats.update(actual_class, path_analysis_time, path_find_time)
    return total_stats

# ===========================================================================

def no_pred_time_budget(args, path_info_list, pred_list, wait_filter):
    # progress_time_budget = dict() # time_budget_ind
    progress_percent_timebudget = dict()
    progress_percent_satpath = dict()
    percent_timebudget_ind = percent_satpath_ind = 0

    random_filter, pred_filter, depth_filter, timeout = wait_filter
    stats = TimeBudgetStats(wait_filter)

    total_stats = get_total_stats(path_info_list, pred_list)
    for actual_class, predicted_class, pred_val, path_analysis_time, path_find_time in path_time_pred(path_info_list, pred_list):
        stats.add_find_time(path_find_time, actual_class)

        sat_rank = get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val)
        if timeout > 0:
            wait_finished = stats.perform_wait(actual_class, path_analysis_time, sat_rank)
            if not wait_finished:
                stats.skip(actual_class, path_analysis_time, sat_rank)
        else:
            stats.add_analysis_time(actual_class, path_analysis_time)
        # time_budget_ind = update_time_budget_stats(progress_time_budget, time_budget_ind, stats, total_stats)
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    stats.check_final_analysis_path(total_stats.num_total_paths)
    stats.check_final_analysis_time(total_stats.total_analysis_time, total_stats.total_find_time, has_skipped_paths=(timeout > 0))
    stats.check_final_finish_notfinish()

    # update_final_budget_stats(progress_time_budget, stats, total_stats)
    update_final_budget_stats(progress_percent_timebudget, stats, total_stats)
    update_final_budget_stats(progress_percent_satpath, stats, total_stats)

    return progress_percent_timebudget, progress_percent_satpath, total_stats

def hard_pred_time_budget(args, path_info_list, path_time_list,
                          pred_list, wait_filter):
    # progress_time_budget = dict() # time_budget_ind
    progress_percent_timebudget = dict()
    progress_percent_satpath = dict()
    percent_timebudget_ind = percent_satpath_ind = 0

    random_filter, pred_filter, depth_filter, wait_time = wait_filter
    stats = TimeBudgetStats(wait_filter)

    total_stats = get_total_stats(path_info_list, pred_list)
    i = 0
    for actual_class, predicted_class, pred_val, path_analysis_time, path_find_time in path_time_pred(path_info_list, pred_list):
        curr_find_time, curr_analysis_time, path_featcollect_time = path_time_list[i]
        i += 1
        assert(curr_find_time == path_find_time)
        assert(curr_analysis_time == path_analysis_time)
        del curr_find_time, curr_analysis_time
        stats.add_find_time(path_find_time, actual_class)

        sat_rank = get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val)
        if wait_time > 0:
            wait_finished = stats.perform_wait(actual_class, path_analysis_time, sat_rank)
        else:
            wait_finished = False

        if not wait_finished:
            stats.add_strategy_overhead(path_featcollect_time)
            stats.add_strategy_overhead(MODEL_PREDICTION_COST)
            sat_rank = get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val)
            predict_analyze = (sat_rank >= args.hard_pred_confidence_threshold)
            if predict_analyze:
                # Keep waiting, the wait unfinished time is part of the analysis time
                stats.long_analyze(actual_class, path_analysis_time, sat_rank)
                stats.use_wait_not_finished(actual_class)
            else:
                stats.skip(actual_class, path_analysis_time, sat_rank)
            stats.count_timegroup_sat_unsat_pred(path_analysis_time, actual_class, predict_analyze)

        stats.check_wait_consistent()
        # time_budget_ind = update_time_budget_stats(progress_time_budget, time_budget_ind, stats, total_stats)
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    stats.check_final_analysis_path(total_stats.num_total_paths)
    stats.check_final_analysis_time(total_stats.total_analysis_time, total_stats.total_find_time, has_skipped_paths=True)
    stats.check_final_finish_notfinish()

    # update_final_budget_stats(progress_time_budget, stats, total_stats)
    update_final_budget_stats(progress_percent_timebudget, stats, total_stats)
    update_final_budget_stats(progress_percent_satpath, stats, total_stats)

    return progress_percent_timebudget, progress_percent_satpath, total_stats

# =================================================================

def pred_rank_time_budget(args, path_info_list, path_time_list,
                          pred_list, wait_filter):
    # progress_time_budget = dict() # time_budget_ind
    progress_percent_timebudget = dict()
    progress_percent_satpath = dict()
    percent_timebudget_ind = percent_satpath_ind = 0

    random_filter, pred_filter, depth_filter, wait_time = wait_filter
    stats = TimeBudgetStats(wait_filter)

    worklist = []
    num_state_stats_map = dict()
    def push_worklist(rank, state_stats):
        state_hash = get_state_hash(num_state_stats_map)
        num_state_stats_map[state_hash] = state_stats

        rank_data = (rank, state_hash)
        if args.mode == "SAT_RANK":
            heappush(worklist, rank_data)
        else:
            assert(args.mode == "RANDOM_RANK")
            worklist.append(rank_data)

    def pop_worklist():
        assert(len(worklist) > 0)
        if args.mode == "SAT_RANK":
            rank, state_hash = heappop(worklist)
        else:
            assert(args.mode == "RANDOM_RANK")
            rand_ind = random.randint(0, len(worklist) - 1)
            rank, state_hash = worklist.pop(rand_ind)
        assert(state_hash in num_state_stats_map)
        state_stats = num_state_stats_map[state_hash]
        del num_state_stats_map[state_hash]
        return rank, state_stats

    total_stats = get_total_stats(path_info_list, pred_list)
    i = 0
    for actual_class, predicted_class, pred_val, path_analysis_time, path_find_time in path_time_pred(path_info_list, pred_list):
        curr_find_time, curr_analysis_time, path_featcollect_time = path_time_list[i]
        i += 1
        assert(curr_find_time == path_find_time)
        assert(curr_analysis_time == path_analysis_time)
        del curr_find_time, curr_analysis_time

        stats.add_find_time(path_find_time, actual_class)

        sat_rank = get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val)
        if wait_time > 0:
            wait_finished = stats.perform_wait(actual_class, path_analysis_time, sat_rank)
        else:
            wait_finished = False

        if not wait_finished:
            path_res = [actual_class, predicted_class, pred_val, path_analysis_time, path_find_time]
            sat_rank = get_sat_rank_with_shift(args, actual_class, predicted_class, pred_val)
            stats.count_timegroup_sat_unsat_pred(path_analysis_time, actual_class, sat_rank >= 0.5)

            # No dependency between paths; Can to feature collection
            # when only model pred is necessary
            stats.add_strategy_overhead(path_featcollect_time)
            stats.add_strategy_overhead(MODEL_PREDICTION_COST)
            if wait_time > 0:
                stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_COST) # save state

            # Filter / analyze if needed
            filter_passed = stats.perform_filter_analyze(sat_rank, actual_class, path_analysis_time)

            if not filter_passed:
                push_worklist(-sat_rank, path_res)
                stats.add_strategy_overhead(PUSH_POP_SAVE_LIST_COST)
                if len(worklist) >= args.list_size:
                    sat_rank, path_res = pop_worklist()
                    stats.add_strategy_overhead(PUSH_POP_SAVE_LIST_COST)
                    if wait_time > 0:
                        stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_COST) # restore state
                    sat_rank = - sat_rank
                    actual_class, predicted_class, pred_val, path_analysis_time, path_find_time = path_res
                    stats.long_analyze(actual_class, path_analysis_time, sat_rank)

                    # Assume we can save / restore states for constraint solving
                    # Wait not finished would be restored, so wait time count as
                    # part of the analysis time
                    assert(path_analysis_time >= wait_time)
                    stats.use_wait_not_finished(actual_class)
        
        stats.check_wait_consistent()
        stats.update_worklist_size(len(worklist))
        # time_budget_ind = update_time_budget_stats(progress_time_budget, time_budget_ind, stats, total_stats)
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    while len(worklist) > 0:
        sat_rank, path_res = pop_worklist()
        stats.add_strategy_overhead(PUSH_POP_SAVE_LIST_COST)
        if wait_time > 0:
            stats.add_strategy_overhead(ANALYSIS_SAVE_RESTORE_COST) # restore state
        sat_rank = - sat_rank
        actual_class, predicted_class, pred_val, path_analysis_time, path_find_time = path_res
        stats.long_analyze(actual_class, path_analysis_time, sat_rank)

        # Assume we can save / restore states for constraint solving
        # Wait not finished would be restored, so wait time count as 
        # part of the analysis time
        assert(path_analysis_time >= wait_time)
        stats.use_wait_not_finished(actual_class)

        stats.check_wait_consistent()
        stats.update_worklist_size(len(worklist))
        # time_budget_ind = update_time_budget_stats(progress_time_budget, time_budget_ind, stats, total_stats)
        percent_timebudget_ind = update_time_budget_stats(progress_percent_timebudget, percent_timebudget_ind, stats, total_stats, is_percent=True)
        percent_satpath_ind = update_time_budget_stats(progress_percent_satpath, percent_satpath_ind, stats, total_stats, is_path_percent=True)

    assert(total_stats.num_total_sat_paths == stats.num_sat_analyzed)
    assert(total_stats.num_total_paths == stats.num_sat_analyzed + stats.num_unsat_analyzed)
    stats.check_final_analysis_path(total_stats.num_total_paths)
    stats.check_final_analysis_time(total_stats.total_analysis_time, total_stats.total_find_time, has_skipped_paths=False)
    stats.check_final_finish_notfinish()
    stats.check_all_notfinished_is_used()

    stats.update_worklist_size(len(worklist))
    # update_final_budget_stats(progress_time_budget, stats, total_stats)
    update_final_budget_stats(progress_percent_timebudget, stats, total_stats)
    update_final_budget_stats(progress_percent_satpath, stats, total_stats)

    return progress_percent_timebudget, progress_percent_satpath, total_stats # progress_time_budget

#===============================================================================

def guide_app_progress_stats(args, app_info_dict, app_pred_dict, app_path_time):
    app_progress_time_budget = dict()
    wait_filter = parse_wait_filter(args.wait_filter_str, has_dep=False)
    for app_name, path_info_list, path_time_list, pred_list in get_info_filtered_apps(app_info_dict, app_path_time, app_pred_dict):
        if args.mode == "NO_PRED":
            random_filter, pred_filter, depth_filter, wait_time = wait_filter
            assert(random_filter is None)
            assert(pred_filter is None)
            progress_stats = no_pred_time_budget(args, path_info_list, pred_list, wait_filter)
        elif args.mode == "HARD_PRED":
            random_filter, pred_filter, depth_filter, wait_time = wait_filter
            assert(random_filter is None)
            assert(pred_filter is None)
            progress_stats = hard_pred_time_budget(args, path_info_list, path_time_list, pred_list, wait_filter)
        elif args.mode == "SAT_RANK" or args.mode == "RANDOM_RANK":
            progress_stats = pred_rank_time_budget(args, path_info_list, path_time_list, pred_list, wait_filter)
        else:
            print("!!! Unrecognized mode: {0} !!!".format(args.mode))
            sys.exit(1)
        app_progress_time_budget[app_name] = progress_stats

        print(".", end="", flush=True)
    print()
    return app_progress_time_budget

#===============================================================================

def run_guide_app_progress_split(guide_app_progress_args, out_filename):
    print("out_filename: {0}".format(out_filename))

    if not os.path.isfile(out_filename):
        app_progress_time_budget = guide_app_progress_stats(**guide_app_progress_args)
        safe_pickle(app_progress_time_budget, out_filename)
    else:
        print("Out file {0} already exits; Skipping...".format(out_filename))
    print()

#===============================================================================

if __name__ == "__main__":
    args = parse_args(has_path_dep=False)

    #
    app_info_dict = load_pickle_res(args.app_info_filename)
    app_path_time = load_pickle_res(args.app_path_time_filename)
    app_pred_dict = load_pickle_res(args.app_pred_filename)

    #
    sig_str = args.mode
    if args.hard_pred_skip_unknown:
        sig_str += "_SKIPUNKNOWN"
    if args.mode == 'SAT_RANK' or args.mode == 'RANDOM_RANK':
        sig_str += "_LISTSIZE{0}".format(args.list_size)
    sig_str += "_{0}_hardpredconf{1:.5f}".format(args.wait_filter_str, args.hard_pred_confidence_threshold)
    if args.shift_model_sat_unsat_threshold is not None:
        sig_str += "_shiftsatunsatthreshold{0:.5f}".format(args.shift_model_sat_unsat_threshold)
    out_filename = "{0}_{1}.p".format(args.out_basename, sig_str)

    #
    guide_app_progress_args = { "args": args,
                                "app_info_dict": app_info_dict,
                                "app_pred_dict": app_pred_dict,
                                "app_path_time": app_path_time }
    run_guide_app_progress_split(guide_app_progress_args, out_filename)

